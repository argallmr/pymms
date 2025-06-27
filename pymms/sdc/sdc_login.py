#!/usr/bin/python
import requests
import urllib3
from urllib.parse import parse_qs

class SDC_Login():

    def __init__(self, username=None, password=None):

        self._sdc_home  = 'https://lasp.colorado.edu/mms/sdc'
        self._info_type = 'download'

        self._session = requests.Session()
        if (username is not None) and (password is not None):
            self._session.auth = (username, password)

    def check_response(self, response):
        '''
        Check the status code for a requests response and perform
        and appropriate action (e.g. log-in, raise error, etc.)

        Parameters
        ----------
        response : `requests.response`
            Response from the SDC

        Returns
        -------
        r : `requests.response`
            Updated response
        '''

        # OK
        if response.status_code == 200:
            r = response

        # Authentication required
        elif response.status_code == 401:
            print('Log-in Required')

            maxAttempts = 4
            nAttempts = 1
            while nAttempts <= maxAttempts:
                # First time through will automatically use the
                # log-in information from the config file. If that
                # information is wrong/None, ask explicitly
                if nAttempts > 1:
                    self.credentials()

                # Remake the request
                #   - Ideally, self._session.send(response.request)
                #   - However, the prepared request lacks the
                #     authentication data
                if response.request.method == 'POST':
                    query = parse_qs(response.request.body)
                    r = self._session.post(response.request.url, data=query)
                else:
                    r = self._session.get(response.request.url)

                # Another attempt
                if r.ok:
                    break
                else:
                    print('Incorrect username or password. {0} tries '
                          'remaining.'.format(maxAttempts-nAttempts))
                    nAttempts += 1

            # Failed log-in
            if nAttempts > maxAttempts:
                raise ConnectionError('Failed log-in.')

        else:
            raise ConnectionError(response.reason)

        # Return the resulting request
        return r

    def get(self):
        '''
        Retrieve information from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(query=False)

        # Check on query
        #   - Use POST if the URL is too long
        r = self._session.get(url, params=self.query())
        if r.status_code == 414:
            r = self._session.post(url, data=self.query())

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r

    def credentials(self, username=None, password=None):
        '''
        Set Log-In credentials for the SDC

        Parameters
        ----------
        username (str):     Account username
        password (str):     Account password
        '''

        # Ask for inputs
        if username is None:
            username = input('username: ')

        if password is None:
            password = input('password: ')

        # Save credentials
        self._session.auth = (username, password)

    def post(self):
        '''
        Retrieve data from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(query=False)

        # Check on query
        r = self._session.post(url, data=self.query())

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r


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