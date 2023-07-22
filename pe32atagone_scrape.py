#!/usr/bin/env python3
"""
ATAG One logging scrape
Get logging data from your ATAG One

Usage::

    $ cat > config.yaml < EOF
    login:
      Email: my.email@login.com
      Password: cGFzc3dvcmQK
    database:
      dsn:
        host: dbhost
        user: dbuser
        database: dbname
        password: cGFzc3dvcmQK
    deviceId: 6808-....-...._..-..-...-...
    EOF

    $ python3 pe32atagone_scrape.py unittest
    ......
    Ran 6 tests in 0.001s

    $ python3 pe32atagone_scrape.py graph
    [(datetime.datetime(2020, 11, 28, 20, 0, tzinfo=<UTC>), 20.1),
     (datetime.datetime(2020, 11, 28, 21, 0, tzinfo=<UTC>), 19.7),
     (datetime.datetime(2020, 11, 28, 22, 0, tzinfo=<UTC>), 19.5),
    ...

    $ python3 pe32atagone_scrape.py diagnostics
    {'averageOutsideTemperature': 6.9,
     'boilerHeatingFor': '',
     'burningHours': 1818.1,
    ...

    $ python3 pe32atagone_scrape.py insert
    (insert stuff into db)

This is work in progress. The goal is to scrape various logging items
from the portal and store them in a database. The hard part
(fetching/decoding the log items) has been completed with the
QuickAndDirtyJavaScriptParser found below.

TODO:
- Add license, year, docs, copyright
- Add logo ref and mention that we're not affiliated
- Extract values from series in usable fashion.
- Document all possible ways to call this.
- Remove duplicate code. And add tests for more content.
- Auto-cleanup cached files once we have the data
TODO2:
- Read https://github.com/kozmoz/atag-one-api/blob/
  5c01e3216003e5f09d74d4dcc128bdc8880c7441/src/main/java/org/juurlink/
  atagone/AtagOneRemoteConnector.java#L109-L132
"""
import base64
import json
import math  # for nan (NaN) and inf (Infinity)
import os
import re
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timedelta
from unittest import TestCase, main as unittest_main

import pytz
import requests
import yaml

try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Temp fix while intermediates on Atag portal are broken. Specify the
# intermediate + root ourself.
if False:
    SESS_KWARGS = {'verify': os.path.join(
        os.path.dirname(__file__), 'portal.atag-one.com.chain')}
else:
    SESS_KWARGS = {}

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
                ident_name = text[(pos + test.v.start()):(pos + test.v.end())]
                if ident_name == 'Infinity':
                    ident = math.inf
                elif ident_name == 'NaN':
                    ident = math.nan
                else:
                    ident = cls.JS_IDENTIFIER(ident_name)
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

    def test_nan_infinity(self):
        source = '''
        {
          name: 'Boiler heating time for hot water',
          tooltip: { valueSuffix: "%", },
          yAxis: 1,
          type: 'column',
          step: 'right',
          data: [
            [Date.UTC(2020,11,19,11,0),0],[Date.UTC(2020,11,19,12,0),0],
            [Date.UTC(2020,11,19,13,0),0],[Date.UTC(2020,11,19,14,0),0],
            [Date.UTC(2020,11,19,15,0),0],[Date.UTC(2020,11,19,16,0),0],
            [Date.UTC(2020,11,19,17,0),0],[Date.UTC(2020,11,19,18,0),0],
            [Date.UTC(2020,11,19,19,0),0],[Date.UTC(2020,11,19,20,0),4],
            [Date.UTC(2020,11,19,21,0),0],[Date.UTC(2020,11,19,22,0),0],
            [Date.UTC(2020,11,19,23,0),0],[Date.UTC(2020,11,20,0,0),0],
            [Date.UTC(2020,11,20,1,0),0],[Date.UTC(2020,11,20,2,0),0],
            [Date.UTC(2020,11,20,3,0),0],[Date.UTC(2020,11,20,4,0),0],
            [Date.UTC(2020,11,20,5,0),0],[Date.UTC(2020,11,20,6,0),0],
            [Date.UTC(2020,11,20,7,0),0],[Date.UTC(2020,11,20,8,0),0],
            [Date.UTC(2020,11,20,9,0),0],[Date.UTC(2020,11,20,10,0),0],
            [Date.UTC(2020,11,20,11,0),0],[Date.UTC(2020,11,20,12,0),0],
            [Date.UTC(2020,11,20,13,0),0],[Date.UTC(2020,11,20,14,0),25],
            [Date.UTC(2020,11,20,15,0),17.8571428571429],
              [Date.UTC(2020,11,20,16,0),26.9230769230769],
            [Date.UTC(2020,11,20,17,0),3.33333333333333],
              [Date.UTC(2020,11,20,18,0),0],
            [Date.UTC(2020,11,20,19,0),0],
              [Date.UTC(2020,11,20,20,0),6.89655172413793],
            [Date.UTC(2020,11,20,21,0),0],[Date.UTC(2020,11,20,22,0),0],
            [Date.UTC(2020,11,20,23,0),0],[Date.UTC(2020,11,21,0,0),0],
            [Date.UTC(2020,11,21,1,0),0],[Date.UTC(2020,11,21,2,0),NaN],
            [Date.UTC(2020,11,21,3,0),NaN],[Date.UTC(2020,11,21,4,0),NaN],
            [Date.UTC(2020,11,21,5,0),NaN],
              [Date.UTC(2020,11,21,6,0),Infinity],
            [Date.UTC(2020,11,21,7,0),NaN],[Date.UTC(2020,11,21,8,0),NaN],
            [Date.UTC(2020,11,21,9,0),0],[Date.UTC(2020,11,21,10,0),0],
            [Date.UTC(2020,11,21,11,0),0]
          ]
        }
        '''
        element, leftovers = QuickAndDirtyJavaScriptParser.parse(source)
        data = element['data']
        self.assertEqual(
            data[-16], ['Date.UTC(2020, 11, 20, 20, 0)', 6.89655172413793])
        self.assertEqual(
            data[-9], ['Date.UTC(2020, 11, 21, 3, 0)', math.nan])
        self.assertEqual(
            data[-6], ['Date.UTC(2020, 11, 21, 6, 0)', math.inf])


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
        resp = sess.get(login_url, **SESS_KWARGS)
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
        resp = sess.get(log_url, **SESS_KWARGS)
        if (resp.status_code != 200 or
                'Latest report time' not in resp.text):
            device_id = None
            sess.cookies.clear()  # wipe stale cookies
    if device_id is None:
        login(sess, load_config_yaml())
        device_id = device_id_from_cookies(sess.cookies)
        log_url = f'{BASE_URL}/Device/LatestReport?deviceId={device_id}'
        resp = sess.get(log_url, **SESS_KWARGS)
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
        resp = sess.get(log_url, **SESS_KWARGS)
        if (resp.status_code != 200 or
                'Central heating water pressure' not in resp.text):
            device_id = None
            sess.cookies.clear()  # wipe stale cookies
    if device_id is None:
        login(sess, load_config_yaml())
        device_id = device_id_from_cookies(sess.cookies)
        log_url = f'{BASE_URL}/Device/GraphTimeSeries?deviceId={device_id}'
        resp = sess.get(log_url, **SESS_KWARGS)
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
        dst_next = None
        for row in data:
            assert (
                len(row) == 2 and row[0].name == 'Date.UTC' and
                len(row[0].args) == 5 and
                isinstance(row[1], (int, float))), row
            a = row[0].args
            # It's not actually Date.UTC, even though it pretends it is.
            # Also: the JS date starts with month 0.
            # We add 1 hour, now 10:00 means "the average between 09:00
            # and 10:00". (This is wrong for fixed times like the room
            # setpoint, but for the other averages it is fine.)
            # FIXME: we should move this parsing to a optional
            # callback-handler in the QuickAndDirtyJavaScriptParser.
            assert 0 <= a[3] <= 23 and a[4] == 0, row
            naive_dt = (
                datetime(a[0], a[1] + 1, a[2], a[3], a[4]))
            # Times during DST change:
            # ['Date.UTC(2021, 2, 28, 0, 0)', 1.7] (2021, 2, 28, 0, 0)
            # ['Date.UTC(2021, 2, 28, 1, 0)', 1.7] (2021, 2, 28, 1, 0)
            # ['Date.UTC(2021, 2, 28, 3, 0)', 1.7] (2021, 2, 28, 3, 0)
            # definitive proof that it is not UTC, and that we add _one_
            # hour after converting to UTC because it signifies "the
            # last time that includes this average"
            try:
                local_dt = TIMEZONE.localize(naive_dt, is_dst=None)
            except pytz.exceptions.AmbiguousTimeError:
                # Whatever.. this only happens when going backward.
                # When going forward there is no ambiguity.
                if dst_next is None:
                    local_dt = TIMEZONE.localize(naive_dt, is_dst=True)
                    dst_next = False
                else:
                    assert dst_next is False, dst_next
                    local_dt = TIMEZONE.localize(naive_dt, is_dst=False)
            local_dt += timedelta(hours=1)
            utc_dt = local_dt.astimezone(pytz.utc)
            # Check time order and add.
            if dt0 is None:
                tdelta = 3600
            else:
                tdelta = (utc_dt - dt0).total_seconds()
            if tdelta == 3600:
                pass  # sorted and hourly
            elif tdelta > 0 and (tdelta % 3600) == 0:
                warnings.warn(f'Missing an hour at {utc_dt} (delta {tdelta})')
            else:
                assert False, (dt0, utc_dt, tdelta)  # unsorted or non-hour
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


def extract_diagnostics(html):
    def name_to_camel(s):
        parts = s.strip().lower().split()
        parts[1:] = [i.capitalize() for i in parts[1:]]
        return ''.join(parts)

    def clean_value(s):
        s = s.strip()
        if s.endswith('&#176;'):  # DEGREE (CELCIUS)
            s = s[0:-6]
        if all(i.isdigit() for i in s.split('.', 1)):
            s = float(s)
        elif s.startswith('20') and len(s) == 19:
            # '2021-04-05 18:16:15' for latestReportTime (in always-CET(!))
            dt, tm = s.split(' ')
            dt = [int(i) for i in dt.split('-')]
            tm = [int(i) for i in tm.split(':')]
            # # time is off by a lot: '2021-04-05 18:16:15' found at 18:00 CEST
            # cet_dt = (
            #     datetime(*(dt + tm), tzinfo=pytz.utc) - timedelta(hours=3))
            naive_dt = datetime(*(dt + tm), tzinfo=None)
            local_dt = TIMEZONE.localize(naive_dt, is_dst=None)
            utc_dt = local_dt.astimezone(pytz.utc)
            # # Date ~is~ was off by 12 minutes or so...
            # # "2021-04-06 12:04:59" is in fact "2021-04-06 11:53:19"
            # # (or earlier)
            # utc_dt -= timedelta(minutes=12)
            # At "2021-04-06 16:26:21" or so, this was fixed. And time
            # was displayed in CEST.
            s = utc_dt
        return s

    pat = re.compile(
        r'<label[^>]*>([^<]*)<\/label>(\s*|(?!<label)<[^>]*>)*([^<]+)')
    res = pat.findall(html, re.DOTALL)
    d = {}
    for name, filler, value in res:
        key = name_to_camel(name)
        assert key not in d, (key, d)
        d[key] = clean_value(value)

    return d


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


def insert_diagnostics_into_db():
    """
    Use diagnostics and insert into DB. Run once per hour.
    """
    def execute_or_ignore(c, q):
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
        except psycopg2.IntegrityError:
            pass

    config = load_config_yaml()
    conn = psycopg2.connect(**config['database']['dsn'])

    temperature_tbl = 'temperature'
    pressure_tbl = 'pressure'

    identifier_to_label_id = {
        'roomTemperature': (temperature_tbl, 1),
        # 'room target temperature': (temperature_tbl, 2),
        'outsideTemperature': (temperature_tbl, 3),
        'chWaterTemperature': (temperature_tbl, 4),
        # XXX:'chReturnTemperature': (temperature_tbl, 4),
        'dhwWaterTemperature': (temperature_tbl, 5),
        # FIXME?'room heating active': (active_tbl, 4),
        # FIXME?'water heating active': (active_tbl, 5),
        'chWaterPressure': (pressure_tbl, 4),
        # 'dt' ?? 'chSetpoint' ? 'dhwSetpoint' ?
    }
    text = fetch_cached_diagnostics_html(clear_cache=True)
    diag = extract_diagnostics(text)

    for identifier, (table, label_id) in identifier_to_label_id.items():
        dt = diag['latestReportTime']
        value = diag[identifier]
        query = (
            f"INSERT INTO {table} (time, location_id, value) VALUES "
            f"('{dt}'::timestamptz, {label_id}, {value});")
        execute_or_ignore(conn, query)


def insert_logging_into_db():
    """
    Use stats/graph and insert into DB. Run once per hour.
    """
    def execute_or_ignore(c, q):
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
        except psycopg2.IntegrityError:
            pass

    config = load_config_yaml()
    conn = psycopg2.connect(**config['database']['dsn'])
    series = extract_series_data(fetch_cached_graph_html(clear_cache=True))

    temperature_tbl = 'temperature'
    active_tbl = 'active'
    pressure_tbl = 'pressure'

    identifier_to_label_id = {
        'room temperature': (temperature_tbl, 1),
        'room target temperature': (temperature_tbl, 2),
        'outside temperature': (temperature_tbl, 3),
        'room heating temperature': (temperature_tbl, 4),
        'water heating temperature': (temperature_tbl, 5),
        'room heating active': (active_tbl, 4),
        'water heating active': (active_tbl, 5),
        'room heating pressure': (pressure_tbl, 4),
    }
    for identifier, (table, label_id) in identifier_to_label_id.items():
        if len(series[identifier].values) < 11:
            print('Unexpected: got only {} values for {}: {}'.format(
                len(series[identifier].values), identifier,
                series[identifier].values))
            continue

        for idx in range(-10, -1):  # insert all but the last one
            dt, value = series[identifier].values[idx]
            query = (
                f"INSERT INTO {table} (time, location_id, value) VALUES "
                f"('{dt}'::timestamptz, {label_id}, {value});")
            execute_or_ignore(conn, query)


def main_diagnostics():
    from pprint import pprint
    text = fetch_cached_diagnostics_html()
    # print(text)
    diag = extract_diagnostics(text)
    # {'averageOutsideTemperature': 0.1,
    #  'boilerHeatingFor': '', # 'DHW' # 'CH'
    #  'burningHours': 1068.1,
    #  'chReturnTemperature': 19.9,
    #  'chSetpoint': 0.0,
    #  'chWaterPressure': 1.5,
    #  'chWaterTemperature': 19.9,
    #  'dhwSetpoint': 60.0,
    #  'dhwWaterTemperature': 28.5,
    #  'dt': 0.0,
    #  'flameStatus': 'Off', # 'On'
    #  'latestReportTime': '2020-11-30 22:25:28',
    #  'outsideTemperature': 6.0,
    #  'roomTemperature': 19.9}
    pprint(diag)


def main_graph():
    from pprint import pprint
    text = fetch_cached_graph_html()
    # print(text)
    series = extract_series_data(text)
    pprint(series['room temperature'].values)


if __name__ == '__main__':
    if sys.argv[1:] == ['unittest']:
        sys.argv.pop()
        unittest_main()
    elif sys.argv[1:] == ['graph']:
        main_graph()
    elif sys.argv[1:] == ['diagnostics']:
        main_diagnostics()
    elif sys.argv[1:] == ['insert']:
        insert_logging_into_db()
        # insert_diagnostics_into_db()
    else:
        assert False
