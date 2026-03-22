import collections
import glob
import itertools
import json
import os
from jsoncomment import JsonComment


def _get_if_not(d, y):
  if y in d:
    return d[y]
  else:
    return None


def clean_json(string):
  json_comment = JsonComment()
  json_obj = json_comment.loads(string)
  json_str = json.dumps(json_obj)
  return json_str


class KeyAuthor(object):
  @staticmethod
  def from_file(key_file):
    assert os.path.exists(key_file), key_file

    key = {}
    with open(key_file) as f:
      json_str = clean_json(f.read())
      if json_str != "":
        key = json.loads(json_str)

    owner = None
    if owner is None:
      owner = _get_if_not(key, "owner")
    if owner is None:
      owner = _get_if_not(key, "account_name")

    key_name = None
    if key_name is None:
      key_name = _get_if_not(key, "name")
    if key_name is None:
      key_name = _get_if_not(key, "key_name")

    access_key = None
    if access_key is None:
      access_key = _get_if_not(key, "access_key")
    if access_key is None:
      access_key = _get_if_not(key, "api_key")

    secret_key = None
    if secret_key is None:
      secret_key = _get_if_not(key, "secret_key")
    if secret_key is None:
      secret_key = _get_if_not(key, "api_secret")

    return KeyAuthor(
        owner=owner,
        key_name=key_name,
        access_key=access_key,
        secret_key=secret_key,
        generated=_get_if_not(key, "generated"),
        generated_human=_get_if_not(key, "generated_human"),
        raw=key,
        key_file=key_file,
    )

  def __init__(self,
               *,
               owner=None,
               key_name=None,
               access_key=None,
               secret_key=None,
               generated=None,
               generated_human=None,
               raw=None,
               key_file=None):
    self._owner = owner
    self._key_name = key_name
    #assert type(access_key) == str
    #assert type(secret_key) == str
    self._access_key = access_key
    self._secret_key = secret_key
    if secret_key is not None:
      self._secret_key_bytes = secret_key.encode("utf-8")
      self._secret_key_upper_bytes = secret_key.upper().encode("utf-8")
    else:
      self._secret_key_bytes = None
      self._secret_key_upper_bytes = None
    self._generated = generated
    self._generated_human = generated_human
    self._raw = raw
    self._key_file = key_file

  @property
  def owner(self):
    return self._owner

  @property
  def key_name(self):
    return self._key_name

  @property
  def access_key(self):
    return self._access_key

  @property
  def secret_key(self):
    return self._secret_key

  @property
  def secret_key_bytes(self):
    return self._secret_key_bytes

  @property
  def secret_key_upper_bytes(self):
    return self._secret_key_upper_bytes

  @property
  def api_key(self):
    return self._access_key

  @property
  def api_secret(self):
    return self._secret_key

  @property
  def key_file(self):
    return self._key_file

  @property
  def refresh_token(self):
    return self._raw.get("refresh_token", None)

  def get_value(self, key):
    # do not add default value
    return self._raw[key]

  def as_json(self):
    o = collections.OrderedDict()
    o["owner"] = self._owner
    o["name"] = self._key_name
    o["access_key"] = self._access_key
    o["secret_key"] = self._secret_key
    if "refresh_token" in self._raw:
      o["refresh_token"] = self._raw["refresh_token"]
    o["generated"] = self._generated
    o["generated_human"] = self._generated_human

    return o

  def as_json_str(self):
    return json.dumps(self.as_json(), indent=2)

  def save_file(self, key_file):
    val = self.as_json_str()
    with open(key_file, "w") as f:
      print(val, file=f)
