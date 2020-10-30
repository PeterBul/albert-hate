import re
import os
import emoji
import sentencepiece as spm
from wordsegment import load, segment
from nltk.tokenize import TweetTokenizer

load()




class FullTokenizer(TweetTokenizer):

  sp_model_epic = os.path.join('albert_base', '30k-clean.model')

  def __init__(self, sp_model, preserve_case=False, reduce_len=False, strip_handles=False, 
               demojize=True, replace_url=True, segment_hashtags=True, 
               correct_user=True, url_to_http=True, remove_url=False, remove_rt=True, change_at=False):
    super().__init__(preserve_case, reduce_len, strip_handles)
    self._demojize = demojize
    self._replace_url = replace_url
    self._segment_hashtags = segment_hashtags
    self._do_correct_user = correct_user
    self._url_to_http = url_to_http
    self._remove_url = remove_url
    self._remove_rt = remove_rt
    self._change_at = change_at
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(sp_model)
    
  
  @classmethod
  def get_sp_model_epic(cls):
    return cls.sp_model_epic
  
  def tokenize(self, tweet):                                                                    
    tweet = self.preprocess(tweet)
    tweet = self.sp_model.EncodeAsPieces(tweet)
    return tweet

  def preprocess(self, tweet):
    tweet = re.sub(r'^!+', '!', tweet)
    if self._do_correct_user:
      tweet = self._correct_user(tweet)
    if self._url_to_http:
      tweet = re.sub(r'(\\bURL\\b)', 'http', tweet)

    tweet = " ".join(super().tokenize(tweet))

    space_pattern = r'\s+'
    giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[#$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    tweet = re.sub(space_pattern, ' ', tweet)
    if self._demojize:
      tweet = self._my_demojize(tweet)
    if self._remove_url:
      tweet = re.sub(giant_url_regex, '', tweet)
    if self._replace_url:
      tweet = re.sub(giant_url_regex, 'http', tweet)
    if self._segment_hashtags:
      tweet = self._segment_tweet(tweet)
    if self._remove_rt:
      rt_p = re.compile(r'\b(?:RT|rt)\b')
      tweet = re.sub(rt_p, '', tweet)
    if self._change_at:
      tweet = re.sub('@', '[at]', tweet)
    return tweet


  def _segment_tweet(self, tweet):
    hashtag_p = re.compile(r'#([\w]+)')
    return re.sub(hashtag_p, lambda matchobj: '# ' + ' '.join(segment(matchobj.group(1))), tweet)
  
  def _correct_user(self, tweet):
    user_p = re.compile(r'((@USER\s*){4,})')
    return re.sub(user_p, '@USER @USER @USER', tweet)

  def _my_demojize(self, string, use_aliases=False, delimiters=(':',':')):

    """Replace unicode emoji in a string with emoji shortcodes. Useful for storage.
    :param string: String contains unicode characters. MUST BE UNICODE.
    :param use_aliases: (optional) Return emoji aliases.  See ``emoji.UNICODE_EMOJI_ALIAS``.
    :param delimiters: (optional) User delimiters other than _DEFAULT_DELIMITER
        >>> import emoji
        >>> print(emoji.emojize("Python is fun :thumbs_up:"))
        Python is fun 👍
        >>> print(emoji.demojize(u"Python is fun 👍"))
        Python is fun :thumbs_up:
        >>> print(emoji.demojize(u"Unicode is tricky 😯", delimiters=("__", "__")))
        Unicode is tricky __hushed_face__
    """

    def replace(match):
        codes_dict = emoji.UNICODE_EMOJI_ALIAS if use_aliases else emoji.UNICODE_EMOJI
        val = codes_dict.get(match.group(0), match.group(0))
        val = val[1:-1]
        val = re.sub('_', ' ', val)
        return delimiters[0] + ' ' + val + ' ' + delimiters[1]

    return re.sub(u'\ufe0f','',(emoji.get_emoji_regexp().sub(replace, string)))


class TweetSpTokenizer(FullTokenizer):
  def __init__(self, preserve_case=False, reduce_len=False, strip_handles=False, 
               demojize=True, replace_url=True, segment_hashtags=True, 
               correct_user=True, url_to_http=True, remove_url=False, remove_rt=True, change_at=True):
    super().__init__(preserve_case, reduce_len, strip_handles, demojize, replace_url, segment_hashtags, 
               correct_user, url_to_http, remove_url, remove_rt, change_at)
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load('/content/drive/My Drive/prosjektoppgave/data/ALBERT/albert-base/assets/30k-clean.model')
  
  def tokenize(self, tweet):
    tweet = super().tokenize(tweet)
    tweet = self.sp_model.EncodeAsPieces(tweet)
    return tweet


