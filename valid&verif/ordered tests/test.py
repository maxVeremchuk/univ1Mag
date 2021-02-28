from formatter import formatter
from formatter import analyzer
import pytest


text_to_format = """if(  ( (a > b ) ) ){{{if(aaa : d  <ddd >){ int a  [  0  ]  =asdf;}}}} aaa   >   =7
{{}
}
"""

text_after_despace = """if(  ( (a  >  b ) ) ){{{if(aaa  :  d   < ddd  > ){ int a  [  0  ]   = asdf;}}}} aaa    >     = 7
{{}
}
"""

text_after_if= """if (  ( (a  >  b ) ) ){{{if (aaa  :  d   < ddd  > ){ int a  [  0  ]   = asdf;}}}} aaa    >     = 7
{{}
}
"""

text_after_for= """if (  ( (a  >  b ) ) ){{{if (aaa  :  d   < ddd  > ){ int a  [  0  ]   = asdf;}}}} aaa    >     = 7
{{}
}
"""

text_after_colon= """if (  ( (a  >  b ) ) ){{{if (aaa :  d   < ddd  > ){ int a  [  0  ]   = asdf;}}}} aaa    >     = 7
{{}
}
"""

text_after_generics= """if (  ( (a  >  b ) ) ){{{if (aaa :  d  <ddd >){ int a  [  0  ]   = asdf;}}}} aaa    >     = 7
{{}
}
"""

text_after_spaces= """if (((a  >  b))) { { {if (aaa :  d  <ddd >) { int a  [0]   = asdf;}}}} aaa    >     = 7
 { {}
}
"""

fmt = formatter.Formatter("test.kt", 4, 80)

@pytest.mark.order(1)
def test_handle_space_constructs():
    line = fmt.handle_space_constructs(text_to_format)
    assert line == text_after_despace
    pytest.shared = line

@pytest.mark.order(2)
def test_handle_if():
    line = fmt.handle_if(pytest.shared)
    assert line == text_after_if
    pytest.shared = line

@pytest.mark.order(3)
def test_handle_for():
    line = fmt.handle_for(pytest.shared)
    assert line == text_after_for
    pytest.shared = line

@pytest.mark.order(4)
def test_handle_colon():
    line = fmt.handle_colon(pytest.shared)
    assert line == text_after_colon
    pytest.shared = line

@pytest.mark.order(5)
def test_handle_generics():
    line = fmt.handle_generics(pytest.shared)
    assert line == text_after_generics
    pytest.shared = line

@pytest.mark.order(6)
def test_handle_spaces():
    line = fmt.handle_spaces(pytest.shared)
    assert line == text_after_spaces

