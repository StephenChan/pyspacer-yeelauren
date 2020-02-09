from typing import List, Tuple, Dict, Union, Optional
from pprint import pformat


class ExtractFeaturesMsg:

    def __init__(self,
                 pk: int,
                 modelname: str,
                 bucketname: str,
                 imkey: str,
                 storage_type: str,
                 rowcols: List[Tuple[int, int]],
                 outputkey: str):

        self.pk = pk
        self.modelname = modelname
        self.bucketname = bucketname
        self.imkey = imkey
        self.storage_type = storage_type
        self.rowcols = rowcols
        self.outputkey = outputkey

    @classmethod
    def deserialize(cls, data: Dict) -> 'ExtractFeaturesMsg':
        msg = ExtractFeaturesMsg(**data)
        msg.rowcols = [tuple(entry) for entry in msg.rowcols]  # JSONstores tuples as lists, we restore it here.
        return msg

    def serialize(self) -> Dict:
        return self.__dict__

    def __repr__(self):
        return pformat(vars(self))

    def __eq__(self, other):
        sd = self.__dict__
        od = other.__dict__
        return sd.keys() == od.keys() and all([sd[key] == od[key] for key in sd])


class ExtractFeaturesReturnMsg:

    def __init__(self,
                 model_was_cashed: bool,
                 runtime: Dict[str, float]):

        self.model_was_cashed = model_was_cashed
        self.runtime = runtime

    @classmethod
    def deserialize(cls) -> 'ExtractFeaturesReturnMsg':
        pass

    def serialize(self) -> Dict:
        pass

    def __str__(self):
        return str(pprint(vars(self)))


class TrainClassifierMsg:

    def __init__(self,
                 pk: str,
                 bucketname: str,
                 traindata: str,
                 model: str,
                 valdata: str):

        self.pk = pk
        self.bucketname = bucketname
        self.traindata = traindata
        self.model = model
        self.valdata = valdata

    @classmethod
    def deserialize(cls) -> 'TrainClassifierMsg':
        pass

    def serialize(self) -> Dict:
        pass


class TrainClassifierReturnMsg:

    def __init__(self):
        pass


class DeployMsg:
    pass


class DeployReturnMsg:
    pass


class TaskMsg:

    def __init__(self,
                 task: str,
                 payload: Union[ExtractFeaturesMsg, TrainClassifierMsg, DeployMsg]):

        self.task = task
        self.payload = payload


class TaskReturnMsg:

    def __init__(self,
                 original_job: TaskMsg,
                 ok: bool,
                 results: Optional[Union[ExtractFeaturesMsg, TrainClassifierReturnMsg, DeployReturnMsg]],
                 error_message: Optional[str]):

        self.original_job = original_job
        self.results = results
        self.ok = ok
        self.error_message = error_message