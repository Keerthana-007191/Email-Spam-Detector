import pandas as pd
import pytest
from sklearn.base import clone

from preprocess import EmailPreprocessor, clean_text


def test_lowercase():
    assert clean_text("HELLO World") == "hello world"


def test_html_tags_removed():
    assert clean_text("<p>Hello <b>World</b></p>") == "hello world"


def test_script_and_style_removed():
    out = clean_text("Free prize <script>alert(1)</script><style>p{x:y}</style>")
    assert out == "free prize"


def test_url_replaced_with_token():
    out = clean_text("Visit http://spam.example.com/win and www.other.com")
    assert out.count("urltoken") == 2
    assert "spam" not in out and "other" not in out


def test_url_inside_href_is_kept_as_token():
    out = clean_text('<a href="http://bad.com/x">Click here</a>')
    assert "urltoken" in out and "click" in out


def test_email_replaced_with_token():
    out = clean_text("Contact john.doe@example.com today")
    assert "emailtoken" in out and "example" not in out


def test_whitespace_normalised():
    assert clean_text("hello    \n\t  world") == "hello world"


def test_punctuation_removed():
    assert clean_text("Hello, world... free-money") == "hello world free money"


@pytest.mark.parametrize("value", ["", "   ", None, float("nan")])
def test_empty_or_missing_input(value):
    assert clean_text(value) == ""


def test_non_string_input_does_not_crash():
    assert clean_text(12345) == "numtoken"
    assert clean_text(b"Hello World") == "hello world"


def test_stopwords_removed_but_negations_kept():
    out = clean_text("This is not a good offer").split()
    assert "not" in out and "good" in out
    assert "this" not in out and "is" not in out


def test_more_negations_kept():
    out = clean_text("I will never go, no way").split()
    assert "never" in out and "no" in out


def test_contractions_keep_negation():
    assert "cannot" in clean_text("I can't do it").split()
    assert "not" in clean_text("we don ' t care").split()  # dataset's spaced style


def test_email_specific_patterns_preserved():
    out = clean_text("Win $1000 now!!!")
    assert "dollartoken" in out and "exclaimtoken" in out and "numtoken" in out


def test_words_with_digits_untouched():
    assert "v1agra" in clean_text("cheap v1agra")


def test_number_masking_can_be_disabled():
    assert clean_text("call 555 1234", mask_numbers=False) == "call 555 1234"


def test_stopword_removal_can_be_disabled():
    assert clean_text("this is good", remove_stopwords=False) == "this is good"


def test_malformed_html_does_not_crash():
    out = clean_text("<div <b>Broken html <i>text")
    assert isinstance(out, str) and "broken" in out and "<" not in out


def test_weird_characters_do_not_crash():
    assert isinstance(clean_text("caf\u00e9 \x00\x01 \u00fc\u00f1\u00ef \U0001F600"), str)


def test_very_long_input_is_truncated():
    assert len(clean_text("word " * 100000, max_chars=100)) <= 100


def test_transformer_accepts_list_and_series():
    pre = EmailPreprocessor()
    texts = ["Hello <b>World</b>", "FREE $$$ now", None]
    assert len(pre.fit_transform(texts)) == 3
    assert pre.transform(pd.Series(texts)) == pre.transform(texts)


def test_transformer_is_deterministic_and_cloneable():
    pre = clone(EmailPreprocessor(mask_numbers=False))
    assert pre.mask_numbers is False
    assert pre.transform(["Same text 42"]) == pre.transform(["Same text 42"])


def test_single_string_input_is_wrapped():
    assert EmailPreprocessor().transform("Hello") == ["hello"]