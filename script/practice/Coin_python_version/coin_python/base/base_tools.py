import requests.auth
import requests
import urllib.parse
import os
import random

class Query(object):
  @staticmethod
  def get_env_http_proxy():
    proxy = None
    env_http_proxy = os.environ.get('http_proxy')
    if env_http_proxy is not None:
      http_proxy_list = env_http_proxy.split(",")
      proxy = random.choice(http_proxy_list)
      if "http://" not in proxy:
        proxy = "http://" + proxy
  
    return proxy
  
  def __init__(self, *, api_host, auth, ip=None, proxies = None):
    self._auth = auth
    self._api_host = api_host
    self._s = requests.Session()
    if proxies is None:
      proxies = {'https': Query.get_env_http_proxy()}
      
    if isinstance(ip, str):
      from requests_toolbelt.adapters import source
      new_source = source.SourceAddressAdapter(ip)
      self._s.mount('http://', new_source)
      self._s.mount('https://', new_source)
      self._s.trust_env = False
    if isinstance(proxies, dict):
      self._s.proxies.update(proxies)

  def query(self, *, method, path, params=None, data=None, headers=None, timeout=None, json=None):
    assert method in ('GET', 'PUT', 'POST', 'DELETE'), method
    url = urllib.parse.urljoin(self._api_host, path)
    response = None

    response = self._s.request(
          method,
          url,
          params=params,
          auth=self._auth,
          timeout=timeout,
          data=data,
          json=json,
          headers=headers)
    return response