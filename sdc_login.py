#!/usr/bin/python
import requests
import urllib3

def parse_form(r):
    '''Parse key-value pairs from the log-in form
    
    Parameters
    ----------
    r (object):    requests.response object.
    
    Returns
    -------
    form (dict):   key-value pairs parsed from the form.
    '''
    # Find action URL
    pstart = r.text.find('<form')
    pend = r.text.find('>', pstart)
    paction = r.text.find('action', pstart, pend)
    pquote1 = r.text.find('"', pstart, pend)
    pquote2 = r.text.find('"', pquote1+1, pend)
    url5 = r.text[pquote1+1:pquote2]
    url5 = url5.replace('&#x3a;', ':')
    url5 = url5.replace('&#x2f;', '/')

    # Parse values from the form
    pinput = r.text.find('<input', pend+1)
    inputs = {}
    while pinput != -1:
        # Parse the name-value pair
        pend = r.text.find('/>', pinput)

        # Name
        pname = r.text.find('name', pinput, pend)
        pquote1 = r.text.find('"', pname, pend)
        pquote2 = r.text.find('"', pquote1+1, pend)
        name = r.text[pquote1+1:pquote2]

        # Value
        if pname != -1:
            pvalue = r.text.find('value', pquote2+1, pend)
            pquote1 = r.text.find('"', pvalue, pend)
            pquote2 = r.text.find('"', pquote1+1, pend)
            value = r.text[pquote1+1:pquote2]
            value = value.replace('&#x3a;', ':')

            # Extract the values
            inputs[name] = value

        # Next iteraction
        pinput = r.text.find('<input', pend+1)
    
    form = {'url': url5,
            'payload': inputs}
    
    return form


def login(username, password):
    '''Log-In to the MMS Science Data Center.
    
    Submit log-in credentials to the MMS Science Data Center team site.

    Parameters:
    -----------
    username (str):     Account username.
    password (str):     Account password.
    
    Returns:
    --------
    Cookies (dict):     Session cookies for continued access to the SDC.
    '''
    
    # Disable warnings because we are not going to obtain certificates
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Attempt to access the site
    #   - Each of the redirects are stored in the history attribute
    url0 = 'https://lasp.colorado.edu/mms/sdc/team/'
    r = requests.get(url0, verify=False)

    # Extract cookies and url
    cookies = r.cookies
    for response in r.history:
        cookies.update(response.cookies)
    
        try:
            url = response.headers['Location']
        except:
            pass
    
    # Submit login information
    payload = {'j_username': username, 'j_password': password}
    r = requests.post(url, cookies=cookies, data=payload, verify=False)

    # After submitting info, we land on a page with a form
    #   - Parse form and submit values to continue
    form = parse_form(r)
    r = requests.post(form['url'], cookies=cookies, data=form['payload'], verify=False)

    # Update cookies to get session information
    cookies = r.cookies
    for response in r.history:
        cookies.update(response.cookies)

    return cookies