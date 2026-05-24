from concurrent.futures import ThreadPoolExecutor

import pytest

from onyx.connectors.models import Document
from onyx.connectors.models import HierarchyNode
from onyx.connectors.web.connector import WEB_CONNECTOR_VALID_SETTINGS
from onyx.connectors.web.connector import WebConnector

EXPECTED_QUOTE = (
    "If you can't explain it to a six year old, you don't understand it yourself."
)


# NOTE(rkuo): we will probably need to adjust this test to point at our own test site
# to avoid depending on a third party site
@pytest.fixture
def quotes_to_scroll_web_connector(request: pytest.FixtureRequest) -> WebConnector:
    scroll_before_scraping = request.param
    connector = WebConnector(
        base_url="https://quotes.toscrape.com/scroll",
        web_connector_type=WEB_CONNECTOR_VALID_SETTINGS.SINGLE.value,
        scroll_before_scraping=scroll_before_scraping,
    )
    return connector


@pytest.mark.parametrize("quotes_to_scroll_web_connector", [True], indirect=True)
def test_web_connector_scroll(quotes_to_scroll_web_connector: WebConnector) -> None:
    all_docs: list[Document] = []
    document_batches = quotes_to_scroll_web_connector.load_from_state()
    for doc_batch in document_batches:
        for doc in doc_batch:
            if isinstance(doc, HierarchyNode):
                continue
            all_docs.append(doc)

    assert len(all_docs) == 1
    doc = all_docs[0]
    assert doc.sections[0].text is not None
    assert EXPECTED_QUOTE in doc.sections[0].text


@pytest.mark.parametrize("quotes_to_scroll_web_connector", [False], indirect=True)
def test_web_connector_no_scroll(quotes_to_scroll_web_connector: WebConnector) -> None:
    all_docs: list[Document] = []
    document_batches = quotes_to_scroll_web_connector.load_from_state()
    for doc_batch in document_batches:
        for doc in doc_batch:
            if isinstance(doc, HierarchyNode):
                continue
            all_docs.append(doc)

    assert len(all_docs) == 1
    doc = all_docs[0]
    assert doc.sections[0].text is not None
    assert EXPECTED_QUOTE not in doc.sections[0].text


MERCURY_EXPECTED_QUOTE = "How can we help?"


@pytest.mark.xfail(
    reason=(
        "flaky. maybe we can improve how we avoid triggering bot protection ormaybe this is just how it has to be."
    ),
)
def test_web_connector_bot_protection() -> None:
    connector = WebConnector(
        base_url="https://support.mercury.com/hc",
        web_connector_type=WEB_CONNECTOR_VALID_SETTINGS.SINGLE.value,
    )
    document_batches = list(connector.load_from_state())
    assert len(document_batches) == 1
    doc_batch = document_batches[0]
    assert len(doc_batch) == 1
    doc = doc_batch[0]
    assert not isinstance(doc, HierarchyNode)
    assert doc.sections[0].text is not None
    assert MERCURY_EXPECTED_QUOTE in doc.sections[0].text


# Salesforce dev docs render the entire article body inside a shadow root
# attached to a <docs-article>-style custom element. `page.content()` only
# serializes the light DOM, so without shadow-DOM flattening only the page
# title and chrome reach BeautifulSoup. This locks in the flattening behavior
# in _get_flattened_html.
def test_web_connector_extracts_shadow_dom_content() -> None:
    connector = WebConnector(
        base_url=(
            "https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta"
            "/apexcode/apex_reserved_words.htm"
        ),
        web_connector_type=WEB_CONNECTOR_VALID_SETTINGS.SINGLE.value,
    )

    all_docs: list[Document] = []
    for batch in connector.load_from_state():
        for doc in batch:
            if isinstance(doc, HierarchyNode):
                continue
            all_docs.append(doc)

    assert len(all_docs) == 1
    text = all_docs[0].sections[0].text
    assert text is not None

    # Body must be substantially longer than the page title alone — guards
    # against regression to title-only extraction.
    assert len(text) > 500, f"extracted text too short ({len(text)} chars): {text!r}"

    # Apex-specific reserved words appear only in the article body, never in
    # the title/breadcrumb chrome.
    for keyword in ("abstract", "transient", "webservice"):
        assert keyword in text.lower(), f"missing body keyword {keyword!r} in: {text!r}"


def test_web_connector_recursive_www_redirect() -> None:
    # Check that https://onyx.app can be recursed if re-directed to www.onyx.app
    # Run in thread pool to avoid conflict with pytest-asyncio's event loop
    def _run_connector() -> list[Document]:
        connector = WebConnector(
            base_url="https://onyx.app",
            web_connector_type=WEB_CONNECTOR_VALID_SETTINGS.RECURSIVE.value,
        )
        return [
            doc
            for batch in connector.load_from_state()
            for doc in batch
            if not isinstance(doc, HierarchyNode)
        ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_connector)
        documents = future.result()

    assert len(documents) > 1
