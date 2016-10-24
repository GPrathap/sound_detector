
import json
import urllib2
import requests

data = {
        'direction': [12, 3, 4, 5, 6],
        'window':[2,3,4,5,6]
}
# req = urllib2.Request('http://192.168.0.4:5000/device/get-direction')
# req.add_header('Content-Type', 'application/json')
# response = urllib2.urlopen(req, json.dumps(data))
#

url = 'http://192.168.0.4:5000/device/get-direction'
headers = {'content-type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)