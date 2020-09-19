from typing import List, Union, Optional, Mapping, Any
import abc

from pyserini.search import JSimpleSearcherResult


__all__ = ['Question', 'Context', 'Reader', 'Answer', 'hits_to_contexts', 'TextType']


TextType = Union['Question', 'Context', 'Answer']


class Question:
    """
    Class representing a question.
    A question contains the question text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The question text.
    id : Optional[str]
        The question id.
    """
    def __init__(self, text: str, id: Optional[str] = None, language: str = "en"):
        self.text = text
        self.id = id
        self.language = language


class Context:
    """
    Class representing a Context to find answer from.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.
    Parameters
    ----------
    text : str
        The context that contains potential answer.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the context. For example, the score might be the BM25 score
        from an initial retrieval stage.
    """

    def __init__(self,
                 text: str,
                 language: str = "en",
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0):
        self.text = text
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score


class Answer:
    """
    Class representing an answer.
    A answer contains the answer text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The answer text.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the answer.
    ctx_score : Optional[float]
        The context score of the answer.
    total_score : Optional[float]
        The aggregated score of answer score and ctx_score
    """
    def __init__(self,
                 text: str,
                 language: str = "en",
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 ctx_score: Optional[float] = 0,
                 total_score: Optional[float] = 0):
        self.text = text
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.ctx_score = ctx_score
        self.total_score = total_score

    def aggregate_score(self, weight):
        self.total_score = weight*self.score + (1-weight)*self.ctx_score



class Reader:
    """
    Class representing a Reader.
    A Reader takes a list Contexts and returns a list of Answer.
    """
    @abc.abstractmethod
    def predict(self, query: Question, texts: List[Context]) -> List[Answer]:
        """
            Find answers from a list of Contexts with respect to a question.
            Parameters
            ----------
            query : Question
                The question.
            texts : List[Context]
                The list of context.
            Returns
            -------
            List[Answer]
                Predicted list of answer.
        """
        pass


def hits_to_contexts(hits: List[JSimpleSearcherResult], field='raw', language="en", blacklist=[]) -> List[Context]:
    """
        Converts hits from Pyserini into a list of texts.
        Parameters
        ----------
        hits : List[JSimpleSearcherResult]
            The hits.
        field : str
            Field to use.
        language : str
            Language of corpus
        blacklist : List[str]
            strings that should not contained
        Returns
        -------
        List[Text]
            List of texts.
     """
    contexts = []
    for i in range(0, len(hits)):
        t = hits[i].raw if field == 'raw' else hits[i].contents
        for s in blacklist:
            if s in t:
                continue
        metadata = {'raw': hits[i].raw, 'docid': hits[i].docid}
        contexts.append(Context(t, language, metadata, hits[i].score))
    return contexts
