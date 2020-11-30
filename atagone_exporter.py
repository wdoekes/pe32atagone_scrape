#!/usr/bin/env python3
"""
ATAG One logging scrape
Get logging data from your ATAG One

Usage::

    $ python3 atagone_exporter.py
    Room temperature
    ['Date.UTC(2020, 10, 27, 0, 0)', 20.0]
    ['Date.UTC(2020, 10, 27, 1, 0)', 19.5]
    ['Date.UTC(2020, 10, 27, 2, 0)', 19.4]
    ...

    $ python3 atagone_exporter.py unittest
    ......
    Ran 6 tests in 0.001s

This is work in progress. The goal is to scrape various logging items
from the portal and store them in prometheus. The hard part
(fetching/decoding the log items) has been completed with the
QuickAndDirtyJavaScriptParser found below.

TODO:
- Add license, year, docs, copyright
- Add logo ref and mention that we're not affiliated
- Extract values from series in usable fashion
- Figure out how we want to export this to prometheus
- Document all possible ways to call this.
- Remove duplicate code. And add tests for more content.
- Auto-cleanup cached files once we have the data
"""
import base64
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from unittest import TestCase, main as unittest_main

import psycopg2
import pytz
import requests
import yaml

BINDIR = os.path.dirname(__file__)
CONFDIR = SPOOLDIR = BINDIR
TIMEZONE = pytz.timezone('Europe/Amsterdam')

# config.yaml, to configure ATAG One API credentials
# > login:
# >   Email: user@domain.tld
# >   Password: YmFzZTY0X2VuY29kZWRfcGFzc3dvcmQ=
CONFIG = os.path.join(CONFDIR, 'config.yaml')
# cookies.js, for temporary cookie storage
COOKIE_JAR = os.path.join(SPOOLDIR, 'cookies.js')
BASE_URL = 'https://portal.atag-one.com'


class set_and_test_idiom:
    """
    Hack to simplify regex match and tests by allowing assignment and
    truth-test in the if-clause.

    Usage::

        test = set_and_test_idiom()

        if test(re.search(r'(cheese|onion)', haystack)):
            crisp_type = test.v.groups()[0]
            print(f'Found special {crisp_type} crisps')
        elif test(re.search(r'(salt|bell pepper)', haystack)):
            crisp_type = test.v.groups()[0]
            print(f'Found regular {crisp_type} crisps')

    Without this, you might resort to something uglier like this::

        m = re.search(r'(cheese|onion)', haystack)
        if m:
            crisp_type = m.groups()[0]
            print(f'Found special {crisp_type} crisps')
        else:
            m = re.search(r'(salt|bell pepper)', haystack)
            if m:
                crisp_type = m.groups()[0]
                print(f'Found regular {crisp_type} crisps')
    """
    __slots__ = ('v',)

    def __call__(self, ret):
        self.v = ret
        return bool(ret)


class SetAndTestIdiomTestCase(TestCase):
    def test_all(self):
        test = set_and_test_idiom()
        with self.assertRaises(AttributeError):
            test.v

        self.assertTrue(test(list(range(3))))
        self.assertEqual(test.v, [0, 1, 2])

        self.assertFalse(test(False))
        self.assertEqual(test.v, False)
        self.assertFalse(test([]))
        self.assertEqual(test.v, [])

        self.assertTrue(test(1))
        self.assertEqual(test.v, 1)


class peek_iter:
    """
    Iterator with peek() and push_back() functionality.

    Usage::

        it = peek_iter(range(9))

        assert next(it) == 0
        assert it.peek() == 1
        assert next(it) == 1

        it.push_back('something else')
        assert next(it) == 'something else'

        for elem in it:
            pass
        # it.push_back()/it.peek() now raise StopIteration
    """
    def __init__(self, iterator):
        self._iterator = iterator
        self._lookahead = []

    def __iter__(self):
        return self

    def __next__(self):
        if self._lookahead:
            return self._lookahead.pop()
        try:
            return next(self._iterator)
        except StopIteration:
            self._lookahead = None
            raise

    def peek(self):
        element = next(self)
        self.push_back(element)
        return element

    def push_back(self, element):
        try:
            self._lookahead.append(element)
        except AttributeError:
            raise StopIteration()


class PeekIterTestCase(TestCase):
    def test_all(self):
        it = peek_iter((i for i in range(9)))
        self.assertEqual(next(it), 0)
        self.assertEqual(it.peek(), 1)
        self.assertEqual(next(it), 1)
        it.push_back('something else')
        self.assertEqual(next(it), 'something else')
        for elem in it:
            pass
        self.assertEqual(elem, 8)
        with self.assertRaises(StopIteration):
            it.peek()
        with self.assertRaises(StopIteration):
            it.push_back('whatever')


class QuickAndDirtyJavaScriptParser:
    """
    Quick and dirty JavaScript parser that parses more than just JSON.

    Example input::

        [{name: 'Room temperature',
          data: [
            [Date.UTC(2020, 10, 26, 17, 0), 20.2],
            [Date.UTC(2020, 10, 26, 18, 0), 20.3]]},
         {name: 'Outside temperature',
          data: [
            [Date.UTC(2020, 10, 26, 17, 0), 14.2],
            [Date.UTC(2020, 10, 26, 18, 0), 13.1]]}]

    When passed to the parse() method get turned into::

        [{"name": "Room temperature",
          "data": [
            [JS_FUNCTION("Date.UTC", (2020, 10, 26, 17, 0)), 20.2],
            [JS_FUNCTION("Date.UTC", (2020, 10, 26, 18, 0)), 20.3]]},
         {"name": "Outside temperature",
          "data": [
            [JS_FUNCTION("Date.UTC", (2020, 10, 26, 17, 0)), 14.2],
            [JS_FUNCTION("Date.UTC", (2020, 10, 26, 18, 0)), 13.1]]}]
    """
    class JS_TOKEN:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f'<{self.name}>'

    class JS_FUNCTION(str):
        def __new__(cls, name, args):
            assert isinstance(name, str)
            assert isinstance(args, tuple)
            value = f'{name}{args!r}'
            obj = str.__new__(cls, value)
            obj.name = name
            obj.args = args
            return obj

    JS_IDENTIFIER = type('JS_IDENTIFIER', (str,), {})
    JS_DICT_BEG = JS_TOKEN('JS_DICT_BEG')
    JS_DICT_END = JS_TOKEN('JS_DICT_END')
    JS_LIST_BEG = JS_TOKEN('JS_LIST_BEG')
    JS_LIST_END = JS_TOKEN('JS_LIST_END')
    JS_PAREN_BEG = JS_TOKEN('JS_PAREN_BEG')
    JS_PAREN_END = JS_TOKEN('JS_PAREN_END')
    JS_COLON = JS_TOKEN('JS_COLON')
    JS_COMMA = JS_TOKEN('JS_COMMA')

    @classmethod
    def parse(cls, text, start=0, end=-1):
        """
        Tokenizes and parses the JavaScript text from start to end.

        If an atom is complete, parsing stops. So, feeding it
        "[1,[2]][3,4]" would return [1,2] as element and return the
        internal iterator containing the tokens for [3,4].

        Right now, the tokenizer has no idea where in the text those
        elements might be, so if you have excess elements, you're on
        your own.

        Usage::

            elem, leftovers = QuickAndDirtyJavaScriptParser.parse(
                '{x:"3",y:{z:func(4)}}')

            elem == {'x': '3', 'y': {'z': JS_FUNCTION('z', (4,))}}
        """
        tokeniter = peek_iter(cls.tokenize(text, start=start, end=end))
        first_token = next(tokeniter)
        root = cls.extract_element(first_token, tokeniter)
        return root, tokeniter

    @classmethod
    def tokenize(cls, text, start=0, end=-1):
        token_dict = {
            '{': cls.JS_DICT_BEG, '}': cls.JS_DICT_END,
            '(': cls.JS_PAREN_BEG, ')': cls.JS_PAREN_END,
            '[': cls.JS_LIST_BEG, ']': cls.JS_LIST_END,
            ':': cls.JS_COLON, ',': cls.JS_COMMA,
        }
        re_identifier = re.compile(
            r'(?s)^[A-Za-z_][A-Za-z0-9_]*(\s*[.]\s*[A-Za-z_][A-Za-z0-9_]*)*')
        re_number = re.compile(r'(?s)^-?[0-9]+([.][0-9]+)?|^-?[.][0-9]+')
        re_string = re.compile(r'(?s)^\'([^\']|\\\\)*\'|^"([^"]|\\\\)*"')
        test = set_and_test_idiom()

        pos = start
        if end == -1:
            end = len(text)

        while pos < end:
            ch = text[pos]
            if ch in ' \t\r\n;':
                pos += 1
            elif ch in token_dict:
                pos += 1
                yield token_dict[ch]
            elif test(re_number.match(text[pos:])):
                num = text[(pos + test.v.start()):(pos + test.v.end())]
                num = float(num) if '.' in num else int(num)
                pos += test.v.end()
                yield num
            elif test(re_identifier.match(text[pos:])):
                ident = cls.JS_IDENTIFIER(
                    text[(pos + test.v.start()):(pos + test.v.end())])
                pos += test.v.end()
                yield ident
            elif test(re_string.match(text[pos:])):
                string = text[(pos + test.v.start()):(pos + test.v.end())]
                string = string[1:-1]  # FIXME: unhandled backslash escapes
                pos += test.v.end()
                yield string
            else:
                sample = '...{}...'.format(text[max(0, pos - 10):(pos + 10)])
                raise ValueError(f'unexpected {ch} in middle of {sample!r}')

    @classmethod
    def extract_element(cls, token, tokeniter):
        if token == cls.JS_LIST_BEG:
            return cls.extract_list(tokeniter)
        if token == cls.JS_DICT_BEG:
            return cls.extract_dict(tokeniter)
        if token == cls.JS_PAREN_BEG:
            return cls.extract_paren(tokeniter)
        if (isinstance(token, cls.JS_IDENTIFIER) and
                tokeniter.peek() == cls.JS_PAREN_BEG):
            next(tokeniter)
            return cls.JS_FUNCTION(token, cls.extract_paren(tokeniter))
        if isinstance(token, (cls.JS_IDENTIFIER, str, float, int)):
            return token
        assert False, token

    @classmethod
    def extract_dict(cls, tokeniter):
        ret = OrderedDict()
        for token in tokeniter:
            if token == cls.JS_DICT_END:
                return ret
            key_atom = cls.extract_element(token, tokeniter)
            assert key_atom is not None, key_atom
            token = next(tokeniter)
            assert token == cls.JS_COLON, token
            token = next(tokeniter)
            value_atom = cls.extract_element(token, tokeniter)
            assert value_atom is not None, value_atom
            ret[key_atom] = value_atom
            token = next(tokeniter)
            if token == cls.JS_DICT_END:
                return ret
            assert token == cls.JS_COMMA, token
        raise EOFError()

    @classmethod
    def extract_list(cls, tokeniter):
        return cls._extract_array(tokeniter, end=cls.JS_LIST_END)

    @classmethod
    def extract_paren(cls, tokeniter):
        return tuple(cls._extract_array(tokeniter, end=cls.JS_PAREN_END))

    @classmethod
    def _extract_array(cls, tokeniter, end=None):
        ret = []
        for token in tokeniter:
            if token == end:
                return ret
            atom = cls.extract_element(token, tokeniter)
            assert atom is not None, atom
            ret.append(atom)
            token = next(tokeniter)
            if token == end:
                return ret
            assert token == cls.JS_COMMA, token
        raise EOFError()


class QuickAndDirtyJavaScriptParserTextCase(TestCase):
    def test_all(self):
        data = '''series: [
            {name: 'Room temperature', tooltip: { valueSuffix: "°C", },
             yAxis: 0, data: [
              [Date.UTC(2020, 10, 26, 17, 0),20.2],
              [Date.UTC(2020, 10, 26, 18, 0),20.3]]
            },
            {name: 'Room setpoint', tooltip: { valueSuffix: "°C" },
             yAxis: 0, data: [
              [Date.UTC(2020, 10, 26, 17, 0),19],
              [Date.UTC(2020, 10, 26, 18, 0),19],]
            }],
          extra_data: "unread"
        '''
        JS_FUNCTION = QuickAndDirtyJavaScriptParser.JS_FUNCTION
        expected = [
            {'name': 'Room temperature', 'tooltip': {'valueSuffix': "°C"},
             'yAxis': 0, 'data': [
                [JS_FUNCTION('Date.UTC', (2020, 10, 26, 17, 0)), 20.2],
                [JS_FUNCTION('Date.UTC', (2020, 10, 26, 18, 0)), 20.3]]},
            {'name': 'Room setpoint', 'tooltip': {'valueSuffix': "°C"},
             'yAxis': 0, 'data': [
                [JS_FUNCTION('Date.UTC', (2020, 10, 26, 17, 0)), 19],
                [JS_FUNCTION('Date.UTC', (2020, 10, 26, 18, 0)), 19]]},
        ]
        element, leftovers = QuickAndDirtyJavaScriptParser.parse(
            data, start=len('series:'))
        self.assertEqual(element, expected)
        self.assertTrue(bool(json.dumps(element)))  # check json.dumps

        JS_COMMA = QuickAndDirtyJavaScriptParser.JS_COMMA
        JS_IDENTIFIER = QuickAndDirtyJavaScriptParser.JS_IDENTIFIER
        self.assertEqual(next(leftovers), JS_COMMA)
        self.assertEqual(next(leftovers), JS_IDENTIFIER('extra_data'))

    def test_early_eof(self):
        with self.assertRaises(StopIteration):
            element, leftovers = QuickAndDirtyJavaScriptParser.parse('[1,[2]')

    def test_exact_eof(self):
        element, leftovers = QuickAndDirtyJavaScriptParser.parse('[1,[2]]')
        with self.assertRaises(StopIteration):
            next(leftovers)
        self.assertEqual(element, [1, [2]])

    def test_also_negative_numbers(self):
        element, leftovers = QuickAndDirtyJavaScriptParser.parse('-1')
        self.assertEqual(element, -1)
        element, leftovers = QuickAndDirtyJavaScriptParser.parse('-.1')
        self.assertEqual(element, -0.1)
        element, leftovers = QuickAndDirtyJavaScriptParser.parse('-0.1')
        self.assertEqual(element, -0.1)

    def test_js_function(self):
        JS_FUNCTION = QuickAndDirtyJavaScriptParser.JS_FUNCTION
        func = JS_FUNCTION('Date.UTC', (1, 2, 'three'))
        self.assertEqual(func.name, 'Date.UTC')
        self.assertEqual(func.args, (1, 2, 'three'))
        self.assertEqual(str(func), "Date.UTC(1, 2, 'three')")
        self.assertEqual(repr(func), "\"Date.UTC(1, 2, 'three')\"")
        self.assertEqual(
            json.dumps({'x': func}), '{"x": "Date.UTC(1, 2, \'three\')"}')


def load_config_yaml():
    """
    Load config.yaml.

    For ATAG One login, it requires:

      login:
        Email: foo@bar.net
        Password: BASE64_ENCODED_PASSWORD

    For push to Postgres/Timescale, also:

      database:
        dsn:
          host: 127.0.0.1
          user: dbuser
          dbname: database
          password: BASE64_ENCODED_PASSWORD
    """
    config = {}
    try:
        with open(CONFIG) as fp:
            config = yaml.safe_load(fp.read())

        # Mandatory.
        assert config.get('login', {}).get('Email')
        assert config.get('login', {}).get('Password')
        config['login']['Password'] = base64.b64decode(
            config['login']['Password']).decode('ascii')
    except Exception as e:
        print(e, file=sys.stderr)
        print('PROBLEM SOURCE:', config, file=sys.stderr)
        raise

    # Base64 decode database.dsn.password.
    db_password = config.get('database', {}).get('dsn', {}).get('password')
    if db_password:
        config['database']['dsn']['password'] = base64.b64decode(
            db_password).decode('ascii')

    return config


def restore_session():
    sess = requests.Session()
    try:
        with open(COOKIE_JAR) as fp:
            jar = json.load(fp)
    except FileNotFoundError:
        pass
    else:
        sess.cookies = requests.cookies.cookiejar_from_dict(jar)
    return sess


def store_session(sess):
    jar = requests.utils.dict_from_cookiejar(sess.cookies)
    with open(COOKIE_JAR, 'w') as fp:
        json.dump(jar, fp)
        fp.write('\n')


def device_id_from_cookies(cookies):
    if 'ATAG' not in cookies:
        return None
    try:
        # {"ATAG": base64({"DeviceId":"1234-5432-1234_54-32-101-123"})}
        js = base64.b64decode(cookies['ATAG']).decode('utf-8')
        device_id = json.loads(js)['DeviceId']
        if not device_id or not re.match('^[0-9_-]+$', device_id):
            raise ValueError()
    except Exception as e:
        print(e, file=sys.stderr)
        print('PROBLEM SOURCE:', cookies, file=sys.stderr)
        raise
    return device_id


def login(sess, config):
    print('Starting login (no existing session?)', file=sys.stderr)
    debug = []
    try:
        login_url = '{}/Account/Login'.format(BASE_URL)
        debug.append(f'GET: {login_url!r}')
        resp = sess.get(login_url)
        debug.append(f'resp: {resp.status_code!r} {sess.cookies!r}')
        debug.append(f'body: {resp.text!r}')
        assert resp.status_code == 200, resp
        # example_text = '''
        #   hod="post"><input name="__RequestVerificationToken"
        #   type="hidden" value="vqM3RhVuyxe5rdKBxuZAZt7itq..." />
        #   <div c'''
        token_input = re.search(
            r'<input\s+[^>]*name=["\']__RequestVerificationToken["\'][^>]*>',
            resp.text)[0]
        # example_token_input = '''
        #   <input name="__RequestVerificationToken" type="hidden"
        #   value="vqM3RhVuyxe5rdKBxuZ" />'''
        token_value = re.search(r'\svalue=["\']([^"\']*)["\']', token_input)
        # example_value = 'vqM3RhVuyxe5rd...'
        value = token_value.groups()[0]
        postdata = (
            [('__RequestVerificationToken', value)] +
            list(config['login'].items()) +
            [('RememberMe', 'true')]  # + [('RememberMe', 'false')]
        )
        debug.append(f'POST {login_url!r} {postdata}')
        resp = sess.post(login_url, data=postdata)  # auto-follow->302->200
        debug.append(f'resp: {resp.status_code!r} {sess.cookies!r}')
        debug.append(f'url: {resp.url!r}')
        debug.append(f'body: {resp.text!r}')
        assert resp.status_code == 200, resp
        assert resp.url == f'{BASE_URL}/', resp.url
        # example_text = '''
        #   ) {\r\n            returnToStart();\r\n        });
        #   \r\n\r\n        $("#mode_settings").on("click", "
        #   .clear_vacation", function () {\r\n
        #   $.get(\'/Home/ClearVacation/
        #   ?deviceId=1234-5432-1234_54-32-101-123\', function
        #   (data) {\r\n'''
        page_device_id = re.search('deviceId=([0-9_-]+)', resp.text)[1]
        cookie_device_id = device_id_from_cookies(sess.cookies)
        assert cookie_device_id == page_device_id, (
            cookie_device_id, page_device_id)
    except Exception as e:
        print(e, file=sys.stderr)
        for line in debug:
            print('DEBUG', line, file=sys.stderr)
        raise
    finally:
        store_session(sess)


def fetch_diagnostics_html():
    sess = restore_session()
    device_id = None
    if sess.cookies:
        device_id = device_id_from_cookies(sess.cookies)
        log_url = f'{BASE_URL}/Device/LatestReport?deviceId={device_id}'
        resp = sess.get(log_url)
        if (resp.status_code != 200 or
                'Latest report time' not in resp.text):
            device_id = None
            sess.cookies.clear()  # wipe stale cookies
    if device_id is None:
        login(sess, load_config_yaml())
        device_id = device_id_from_cookies(sess.cookies)
        log_url = f'{BASE_URL}/Device/LatestReport?deviceId={device_id}'
        resp = sess.get(log_url)
        if (resp.status_code != 200 or
                'Latest report time' not in resp.text):
            raise ValueError((resp, resp.text))
    store_session(sess)
    return resp.text


def fetch_graph_html():
    sess = restore_session()
    device_id = None
    if sess.cookies:
        device_id = device_id_from_cookies(sess.cookies)
        log_url = f'{BASE_URL}/Device/GraphTimeSeries?deviceId={device_id}'
        resp = sess.get(log_url)
        if (resp.status_code != 200 or
                'Central heating water pressure' not in resp.text):
            device_id = None
            sess.cookies.clear()  # wipe stale cookies
    if device_id is None:
        login(sess, load_config_yaml())
        device_id = device_id_from_cookies(sess.cookies)
        log_url = f'{BASE_URL}/Device/GraphTimeSeries?deviceId={device_id}'
        resp = sess.get(log_url)
        if (resp.status_code != 200 or
                'Central heating water pressure' not in resp.text):
            raise ValueError((resp, resp.text))
    store_session(sess)
    return resp.text


def fetch_cached_diagnostics_html(clear_cache=False):
    """
    Return cached diagnostics html.

    The clear_cache is a hack so we flush the cache manually, while
    keeping the caching method internal to this function.

    <legend>DIAGNOSTICS</legend>
    <div class="form-group no-border-top">
      <label class="col-xs-6 control-label">Latest report time</label>
      <div class="col-xs-6">
        <p class="form-control-static">2020-11-30 22:19:22</p>
    ...
    """
    cache_file = os.path.join(SPOOLDIR, 'diagnostics.html')
    try:
        if clear_cache:
            raise FileNotFoundError()  # pretend it wasn't there
        with open(cache_file) as fp:
            text = fp.read()
        if 'Latest report time' not in text:
            raise ValueError()
    except (FileNotFoundError, ValueError):
        text = fetch_diagnostics_html()
        with open(cache_file, 'w') as fp:
            fp.write(text)
    return text


def fetch_cached_graph_html(clear_cache=False):
    """
    Return cached graph html.

    The clear_cache is a hack so we flush the cache manually, while
    keeping the caching method internal to this function.

    <html>...<script...>
    ...
    series: [{
      name: 'Room temperature',
      tooltip: { valueSuffix: "°C", },
      yAxis: 0,
      data: [[Date.UTC(2020, 10, 26, 17, 0),1.9],[...]]
    }]
    ...
    """
    cache_file = os.path.join(SPOOLDIR, 'graph.js.html')
    try:
        if clear_cache:
            raise FileNotFoundError()  # pretend it wasn't there
        with open(cache_file) as fp:
            text = fp.read()
        if '<script' not in text:
            raise ValueError()
    except (FileNotFoundError, ValueError):
        text = fetch_graph_html()
        with open(cache_file, 'w') as fp:
            fp.write(text)
    return text


class AtagOnePortalGraphData:
    # FIXME: these names/labels may need some work: use camelCase
    # instead? like the JS diagnostics?
    NAME_TO_IDENTIFIER = {
        'Room temperature': 'room temperature',
        'Room setpoint': 'room target temperature',
        'Outside temperature': 'outside temperature',
        'Central heating temperature': 'room heating temperature',
        'Hot water temperature': 'water heating temperature',
        'Boiler heating time for CH': 'room heating active',
        'Boiler heating time for hot water': 'water heating active',
        'Central heating water pressure': 'room heating pressure',
    }

    @classmethod
    def from_series_entry(cls, entry):
        identifier = cls.NAME_TO_IDENTIFIER[entry['name']]
        return cls(identifier=identifier, data=entry['data'])

    def __init__(self, identifier, data):
        self.identifier = identifier
        values = []
        dt0 = None
        for row in data:
            assert (
                len(row) == 2 and row[0].name == 'Date.UTC' and
                len(row[0].args) == 5 and
                isinstance(row[1], (int, float))), row
            a = row[0].args
            # It's not actually Date.UTC, even though it pretends it is.
            # Also: the JS date starts with month 0. And we add 30
            # minutes, because it's an average over the entire hour.
            assert a[4] == 0, row
            naive_dt = datetime(a[0], a[1] + 1, a[2], a[3], a[4] + 30)
            local_dt = TIMEZONE.localize(naive_dt, is_dst=None)
            utc_dt = local_dt.astimezone(pytz.utc)
            # Check time order nad add.
            assert dt0 is None or (utc_dt - dt0).total_seconds() == 3600, (
                dt0, utc_dt)  # sorted AND hour-offsets
            values.append((utc_dt, row[1]))  # date, value
            dt0 = utc_dt
        self.values = values

    def get_last_value(self):
        return self.values[-1][1]  # last element; value

    def __iter__(self):
        return iter(self.values)


def extract_series_data(html_with_js):
    # Fetch the "series" dict key with a list of elements ("[").
    m = re.search(r'\sseries\s*:\s*\[', html_with_js, re.DOTALL)
    # Start the parsing at the "[".
    series, leftovers = QuickAndDirtyJavaScriptParser.parse(
        html_with_js, start=(m.end() - 1))

    # Ignore the leftover tokens.. there will be HTML as well :)
    # Check that we got exactly the values we wanted.
    expected_names = [
        # FIXME: this is duplicate code, see AtagOnePortalGraphData
        'Room temperature', 'Room setpoint', 'Outside temperature',
        'Central heating temperature', 'Hot water temperature',
        'Boiler heating time for CH', 'Boiler heating time for hot water',
        'Central heating water pressure']
    received_names = [i['name'] for i in series]
    assert expected_names == received_names, (expected_names, received_names)

    # Convert to something usable.
    # FIXME: this should not be a dict/list, but a full fledged class,
    # which also allows to_json().
    datas = [AtagOnePortalGraphData.from_series_entry(i) for i in series]
    return dict((i.identifier, i) for i in datas)


class SeriesGetter:
    """
    FIXME: This needs a better name.
    """
    def __init__(self):
        self.series = []
        self.time = None
        self.refresh()

    def refresh(self):
        print('Requesting fresh data', file=sys.stderr)
        fresh_data = fetch_cached_graph_html(clear_cache=True)
        self.series = extract_series_data(fresh_data)
        self.time = time.time()

    def get_last(self, identifier):
        # For now, the data is aggregated per hour. So it would make
        # sense to only get a new one after the hour has passed.
        # (time//3600?)
        # FIXME: this is actually buggy, as the last value is an
        # average that keeps changing. By returning this, we're
        # creating a sawtooth where the first minutes of the hour are
        # always more extreme.
        # FIXME: should do the hour-check instead..
        # but instead, we'll want none of the graph stuff here, but the
        # instant-stuff from latestReportTime/diagnostics instead.
        if (time.time() - 900) > self.time:
            self.refresh()

        return self.series[identifier].get_last_value()


def prometheus():
    from prometheus_client import Gauge, start_http_server

    temp_gauge = Gauge(
        'temperature', 'Room, outside, heating temperatures (in C)', ['what'])
    heat_gauge = Gauge(
        'heating', 'Heating of room or water (in percentage)', ['what'])
    press_gauge = Gauge(
        'pressure', 'Central heating water pressure (in Bar)', ['what'])

    getter = SeriesGetter()

    temp_gauge.labels(what='room').set_function((
        lambda: getter.get_last('room temperature')))
    temp_gauge.labels(what='room-target').set_function((
        lambda: getter.get_last('room target temperature')))
    temp_gauge.labels(what='outside').set_function((
        lambda: getter.get_last('outside temperature')))
    temp_gauge.labels(what='boiler-room').set_function((
        lambda: getter.get_last('room heating temperature')))
    temp_gauge.labels(what='boiler-water').set_function((
        lambda: getter.get_last('water heating temperature')))

    heat_gauge.labels(what='boiler-room').set_function((
        lambda: getter.get_last('room heating active')))
    heat_gauge.labels(what='boiler-water').set_function((
        lambda: getter.get_last('water heating active')))

    press_gauge.labels(what='boiler').set_function((
        lambda: getter.get_last('room heating pressure')))

    start_http_server(9002)
    while True:
        time.sleep(60)


def insert_latest_into_db():
    """
    Run once per hour.
    """
    config = load_config_yaml()
    conn = psycopg2.connect(**config['database']['dsn'])
    series = extract_series_data(fetch_cached_graph_html(clear_cache=True))
    identifier_to_label_id = {
        'room temperature': 1,
        'room target temperature': 2,
        'outside temperature': 3,
        'room heating temperature': 4,
        'water heating temperature': 5,
    }
    for identifier, label_id in identifier_to_label_id.items():
        dt, value = series[identifier].values[-2]  # only insert next to last
        query = (
            f"INSERT INTO temperature (time, location_id, value) VALUES "
            f"('{dt}'::timestamptz, {label_id}, {value});")
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(query)  # psycopg2.IntegrityError

    identifier_to_label_id = {
        'room heating active': 4,
        'water heating active': 5,
    }
    for identifier, label_id in identifier_to_label_id.items():
        dt, value = series[identifier].values[-2]  # only insert next to last
        query = (
            f"INSERT INTO active (time, location_id, value) VALUES "
            f"('{dt}'::timestamptz, {label_id}, {value});")
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(query)  # psycopg2.IntegrityError

    identifier_to_label_id = {
        'room heating pressure': 4,
    }
    for identifier, label_id in identifier_to_label_id.items():
        dt, value = series[identifier].values[-2]  # only insert next to last
        query = (
            f"INSERT INTO pressure (time, location_id, value) VALUES "
            f"('{dt}'::timestamptz, {label_id}, {value});")
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(query)  # psycopg2.IntegrityError


def main():
    text = fetch_cached_diagnostics_html()
    # {
    # deviceId: "6808-1401-3109_15-30-001-123",
    # latestReportTime: "2020-11-30 22:31:39",
    # deviceIP: "10.0.100.50",
    # burningHours: 283.91,
    # roomTemperature: 19.9,
    # outsideTemperature: 4.4,
    # dhwSetpoint: 60.0,
    # dhwWaterTemperature: 46.8,
    # chSetpoint: 42.3,
    # chWaterTemperature: 46.6,
    # chWaterPressure: 1.6,
    # chReturnTemperature: 40.3,
    # targetTemperature: 20.0,
    # dhwWaterTemp: 46.8,
    # dhwWaterPres: 0.0,
    # ...
    # }
    print(text)


if __name__ == '__main__':
    if sys.argv[1:] == ['unittest']:
        sys.argv.pop()
        unittest_main()
    elif sys.argv[1:] == ['insert']:
        insert_latest_into_db()
    elif sys.argv[1:] == ['prometheus']:
        prometheus()
    elif sys.argv[1:] == []:
        main()
    else:
        assert False
