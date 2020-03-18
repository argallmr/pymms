#!/usr/bin/python

import sys
import os
import requests
import urllib3

global debug

def myGet(url, expected_status,
          allow_redirects=False, 
          cookies={}):

    global debug

    r = requests.get(url, verify=0,
                     allow_redirects=allow_redirects,
                     cookies=cookies)
#    if debug==True:
#        print('GET ', url)
#        print('status: %d' % r.status_code)
#        print('cookies: ', r.cookies.get_dict())
#        print('headers: ', r.headers)
#        print('text: ', r.text)

    if r.status_code != expected_status:
        print('>>>> !!! Unexpected status: ', r.status_code)
        sys.exit(1)

    return r


def myPost(url, expected_status, payload,
           allow_redirects=False,
           cookies={}):

    global debug

    r = requests.post(url, verify=0, 
                      allow_redirects=allow_redirects,
                      cookies=cookies,
                      data=payload)
#    if debug==True:
#        print('GET ', url)
#        print('status: %d' % r.status_code)
#        print('cookies: ', r.cookies.get_dict())
#        print('headers: ', r.headers)
#        print('text: ', r.text)

    if r.status_code != expected_status:
        print('>>>> !!! Unexpected status: ', r.status_code)
        sys.exit(1)

    return r


def parseResponseText(text):
    pbeg = text.find('<form')
    pend = text.find('>', pbeg)
    paction = text.find('action=', pbeg,      pend)
    pquote1 = text.find('"',       paction,   pend)
    pquote2 = text.find('"',       pquote1+1, pend)

    url = text[pquote1+1:pquote2]
    url = url.replace('&#x3a;', ':')
    url = url.replace('&#x2f;', '/')

    pinput1 = text.find('<input', pbeg)
    pend1   = text.find('/>',     pinput1)
    pname1  = text.find('name',  pinput1,   pend1)
    pquote1 = text.find('"',     pname1,    pend1)
    pquote2 = text.find('"',     pquote1+1, pend1)
    pvalue1 = text.find('value', pquote2+1, pend1)
    pquote3 = text.find('"',     pvalue1,   pend1)
    pquote4 = text.find('"',     pquote3+1, pend1)

    name1 = text[pquote1+1:pquote2]
    #print(name1)
    value1 = text[pquote3+1:pquote4]
    value1 = value1.replace('&#x3a;', ':')
    #print(value1)

    pinput2 = text.find('<input', pquote4)
    pend2   = text.find('/>',     pinput2)
    pname2  = text.find('name',  pinput2,   pend2)
    pquote5 = text.find('"',     pname2,    pend2)
    pquote6 = text.find('"',     pquote5+1, pend2)
    pvalue2 = text.find('value', pquote6+1, pend2)
    pquote7 = text.find('"',     pvalue2,   pend2)
    pquote8 = text.find('"',     pquote7+1, pend2)

    name2 = text[pquote5+1:pquote6]
    value2 = text[pquote7+1:pquote8]

    payload = { name1:value1, name2:value2 }

    return ( url, payload )


def LaspLogin(username, password, quiet=False):
    '''Login to the LASP MMS SDC team site with username and password
       and return a session cookie that can be used for data requests.
    '''

    # Disable warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url0 = 'https://lasp.colorado.edu/mms/sdc/team/'

    #print("Stage 1")
    #r = myGet(url0, 302, allow_redirects=False)
    #url1 = r.headers['Location']

    #print("Stage 2")
    #r = myGet(url1, 302, allow_redirects=False)
    #url2 = r.headers['Location']

    #print("Stage 3")
    #cookies1 = r.cookies
    #r = myGet(url2, 302, allow_redirects=False, cookies=cookies1)
    #url3 = r.headers['Location']

    #print("Stage 4")
    #r = myGet(url3, 200, allow_redirects=False, cookies=cookies1)

    print("Stages 1-4")
    r = myGet(url0, 200, allow_redirects=True)
    cookies1 = r.history[1].cookies
    url3 = r.history[2].headers['Location']


    payload = { 'j_username':username, 'j_password':password }

    #print("Stage 5")
    #r = myPost(url3, 302, payload, 
               #allow_redirects=False, cookies=cookies1)

    #cookies2 = r.cookies
    #all_cookies = cookies1
    #all_cookies.update(cookies2)
    #url4 = r.headers['Location']

    #print("Stage 6")
    #r = myGet(url4, 200, allow_redirects=False, cookies=all_cookies)

    print("Stages 5-6")
    r = myPost(url3, 200, payload, allow_redirects=True, cookies=cookies1)

    (url5, payload) = parseResponseText(r.text)

    print("Stage 7")
    r = myPost(url5, 302, payload, allow_redirects=False)

    session_cookie = r.cookies

    # # Check if we can get to the original SDC page now
    # print("Stage 8")
    # url6 = r.headers['Location']
    # r = myGet(url6, 200, allow_redirects=False, cookies=session_cookie)

    # Return the cookie as a tuple (key,value)
    d = session_cookie.get_dict()
    key_list = list(d.keys())
    key = key_list[0]

    return ( key, d[key] )

    #import code
    #code.interact(local=locals())

# ---------------------------------------------------------------------------

if __name__ == '__main__':

    global debug

    debug = True

    if len(sys.argv) < 3:
        msg = 'ERROR: requiring at least two arguments'
        print(msg, file=sys.stderr)
        msg = 'args: username password [verbose=0] [quiet=0]'
        print(msg, file=sys.stderr)
        sys.exit(1)

    username = sys.argv[1]
    password = sys.argv[2]

    if len(sys.argv) > 3:
        verbose = int(sys.argv[3])
    else:
        verbose = 0

    if len(sys.argv) > 4:
        quiet = int(sys.argv[4])
    else:
        quiet = 0

    try:
        home_dir = os.environ['HOME']
    except KeyError:
        if quiet==False:
            msg = 'ERROR: environment variable HOME is not set'
            print(msg, file=sys.stderr)
        sys.exit(1)

    # Delete existing cookie file
    cookie_file = home_dir + '/lasp_cookie.txt'
    try:
        os.remove(cookie_file)
    except OSError:
        pass

    # Get new session cookie (returned as a tuple (key,value) )
    session_cookie = LaspLogin(username, password, quiet)

    # Save the new cookie
    f = open(cookie_file, 'w')
    f.write(session_cookie[0] + '\n')
    f.write(session_cookie[1] + '\n')
    f.close()
    if verbose==True or debug==True:
        print('Wrote LASP session cookie to ' + cookie_file)
    # ----------------------------------------------------------


