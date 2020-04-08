#!/usr/bin/python

import sys
import os
import requests
import urllib3

def LaspLogin(username, password, quiet=False):
    '''Login to the LASP MMS SDC team site with username and password
       and return a session cookie that can be used for data requests.
    '''

    # Disable warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    url0 = 'https://lasp.colorado.edu/mms/sdc/team/'
    r = requests.get(url0, verify=0, allow_redirects=False)
    #print('GET %s' % url0)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)
    if r.status_code != 302:
        if quiet==False:
            msg = 'Stage 1 error: expecting redirect (status 302)'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        sys.exit(1)
    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    url1 = r.headers['Location']
    r = requests.get(url1, verify=0, allow_redirects=False)
    #print('GET %s' % url1)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)

    if r.status_code != 302:
        if quiet==False:
            msg = 'Stage 2 error: expecting redirect (status 302)'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        sys.exit(1)

    url2 = r.headers['Location']
    cookies1 = r.cookies

    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    r = requests.get(url2, verify=0, allow_redirects=False, cookies=cookies1)
    #print('GET %s' % url2)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)

    if r.status_code != 302:
        if quiet==False:
            msg = 'Stage 3 error: expecting redirect (status 302)'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        sys.exit(1)

    url3 = r.headers['Location']

    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    r = requests.get(url3, verify=0, allow_redirects=False, cookies=cookies1)
    #print('GET %s' % url3)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)
    #print('text: ', r.text)

    if r.status_code != 200:
        if quiet==False:
            msg = 'Stage 4 error: expecting status 200'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    payload = { 'j_username':username, 'j_password':password }
    r = requests.post(url3, verify=0, allow_redirects=False, \
                      cookies=cookies1, data=payload)
    #print('POST %s' % url3)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)
    #print('text: ', r.text)

    if r.status_code != 302:
        if quiet==False:
            msg = 'Stage 5 error: expecting redirect (status 302)'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        # this could be caused by invalid credentials
        # inspect r.text for 'Credentials not recognized'
        sys.exit(1)

    cookies2 = r.cookies
    all_cookies = cookies1
    all_cookies.update(cookies2)
    url4 = r.headers['Location']

    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    r = requests.get(url4, verify=0, allow_redirects=False, cookies=all_cookies)
    print('GET %s' % url4)
    print('cookies: ', all_cookies.get_dict())
    print('status: %d' % r.status_code)
    print('cookies: ', r.cookies.get_dict())
    print('headers: ', r.headers)
    print('text: ', r.text)

    if r.status_code != 200:
        if quiet==False:
            msg = 'Stage 6 error: expecting status 200'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        sys.exit(1)

    pbeg = r.text.find('<form')
    pend = r.text.find('>', pbeg)
    paction = r.text.find('action=', pbeg,      pend)
    pquote1 = r.text.find('"',       paction,   pend)
    pquote2 = r.text.find('"',       pquote1+1, pend)

    url5 = r.text[pquote1+1:pquote2]
    url5 = url5.replace('&#x3a;', ':')
    url5 = url5.replace('&#x2f;', '/')
    print(url5)

    pinput1 = r.text.find('<input', pbeg)
    pend1   = r.text.find('/>',     pinput1)
    pname1  = r.text.find('name',  pinput1,   pend1)
    pquote1 = r.text.find('"',     pname1,    pend1)
    pquote2 = r.text.find('"',     pquote1+1, pend1)
    pvalue1 = r.text.find('value', pquote2+1, pend1)
    pquote3 = r.text.find('"',     pvalue1,   pend1)
    pquote4 = r.text.find('"',     pquote3+1, pend1)

    name1 = r.text[pquote1+1:pquote2]
    print(name1)
    value1 = r.text[pquote3+1:pquote4]
    value1 = value1.replace('&#x3a;', ':')
    print(value1)

    pinput2 = r.text.find('<input', pquote4)
    pend2   = r.text.find('/>',     pinput2)
    pname2  = r.text.find('name',  pinput2,   pend2)
    pquote5 = r.text.find('"',     pname2,    pend2)
    pquote6 = r.text.find('"',     pquote5+1, pend2)
    pvalue2 = r.text.find('value', pquote6+1, pend2)
    pquote7 = r.text.find('"',     pvalue2,   pend2)
    pquote8 = r.text.find('"',     pquote7+1, pend2)

    name2 = r.text[pquote5+1:pquote6]
    print(name2)
    value2 = r.text[pquote7+1:pquote8]
    print(value2)

    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')

    payload = { name1:value1, name2:value2 }
    r = requests.post(url5, verify=0, allow_redirects=False, \
                    data=payload)
    #print('POST %s' % url5)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)

    if r.status_code != 302:
        if quiet==False:
            msg = 'Stage 7 error: expecting redirect (status 302)'
            print(msg, file=sys.stderr)
            msg = 'Actual status: %d' % r.status_code
            print(msg, file=sys.stderr)
        sys.exit(1)

    session_cookie = r.cookies
    url6 = r.headers['Location']

    # ----------------------------------------------------------------
    #print('-----------------------------------------------------------')


    #r = requests.get(url6, verify=0, allow_redirects=False, cookies=session_cookie)
    #print('GET %s' % url6)
    #print('status: %d' % r.status_code)
    #print('cookies: ', r.cookies.get_dict())
    #print('headers: ', r.headers)

    #if r.status_code != 200:
        #print('ERROR: expecting status 200')
        #sys.exit(1)

    #import code
    #code.interact(local=locals())

    # Return the cookie as a tuple (key,value)
    d = session_cookie.get_dict()
    key_list = list(d.keys())
    key = key_list[0]

    return ( key, d[key] )

# ---------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) < 3:
        msg = 'ERROR: requiring at least two arguments'
        print(msg, file=sys.stderr)
        msg = 'args: username password [verbose=0] [quiet=0]'
        print(msg, file=sys.stderr)
        sys.exit(1)

    username = sys.argv[1]
    password = sys.argv[2]
    
    import pdb
    pdb.set_trace()

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
    if verbose==True:
        print('Wrote LASP session cookie to ' + cookie_file)
    # ----------------------------------------------------------


