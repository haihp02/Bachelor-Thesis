import copy
import json
from typing import Annotated, Union

class ModelArgs():
    '''
    Class to hold model 's arguments
    '''
    def __init__(self):
        # Model args
        self.name_or_path: str = None
        self.model_max_length: int
        self.ner_hidden_size = 256
        self.ner_layer = 'last'
        self.embedding_hidden_size = 256
        self.embedding_dim = 128
        # COIL
        self.tok_embedding_dim = 32
        self.cls_embedding_dim = 768
        self.coil_score_type = 'full'

        self.add_special_tokens: bool = False
        self.ner_augment = None
        self.kw_threshold: float = 0
        self.weight_mask: int = -1

        self.query_max_length: int = 32
        self.mask_punctuation: bool = True
        self.similarity_metric: str = 'cosine'
        self.query_pad_token: str = None
        self.doc_marker_token: str = None
        self.query_marker_token: str = None
        self.ir_loss_type = 'mixed'
        self.lora_config: dict = None


class DataArgs():
    '''
    Class to hold datamodule 's arguments
    '''
    def __init__(self):
        # Data args
        self.batch_size: Annotated[Union[int, None], 'Size of the batch'] = None
        self.processed_data_dir: str = None

        self.train_data_path: str = None
        self.val_data_path: str = None
        self.test_data_path: str = None

        self.num_ner_tags: int = 9
        self.dummy_label_id: int = -100
        self.ner_first_token: bool = True
        self.pos_first_token: bool = True
        self.collection_path: str = None
        self.queries_path: str = None
        self.train_triples_path: str = None
        self.val_qrels_path: str = None

        # MLM
        self.mlm_chunk_size = None
        self.concat_mlm_chunks = True
        self.mask_whole_word = True
        self.pre_mask_test: bool = True

        self.dataloader_num_workers: int = 0


class TrainingArgs:
    '''
    Class to hold trainer 's arguments
    '''
    def __init__(self):
        self.resume_from_checkpoint: bool = False
        self.num_train_epochs: int = None
        self.max_train_steps: int = None
        self.scheduler: str = None
        self.lr_warmup_steps: int = None
        self.gradient_accumulation_steps: str = 1
        self.split_batches: bool = True
        self.mixed_precision: str = 'no'
        self.use_8bit_optimizer: bool = False
        self.gradient_checkpointing: bool = False
        self.lr = 1e-5
        self.dropout_rate = 0.05
        self.adam_beta1: float = 0.9
        self.adam_beta2: float = 0.999
        self.adam_eps: float = 1e-8 
        self.weight_decay: float = 0.01
        self.max_grad_norm: float = 1.0
        self.output_dir: str = None
        self.checkpointing_steps: int = None
        self.train_bert: bool = True
        # Logging
        self.log_with = 'wandb'
        self.exp_name = None
        self.run_name = None


class IndexRetrieveArgs:
    '''
    Class to hold indexer and retriever 's arguments
    '''
    def __init__(self):
        self.apply_weight_mask = False
        self.nbits: int = 1
        self.plaid_num_partitions_coef: int = 2
        self.centroid_score_threshold: float
        self.top_k_tokens: int
        self.ncells: int
        self.ndocs: int
        self.post_pruning_threshold: float = 0.5


class Args(ModelArgs, DataArgs, TrainingArgs, IndexRetrieveArgs):
    '''
    Class to hold all arguments, args defined here are general
    '''
    def __init__(self):
        # Init all Args classes
        ModelArgs.__init__(self)
        DataArgs.__init__(self)
        TrainingArgs.__init__(self)
        IndexRetrieveArgs.__init__(self)

        self.seed: int = None
        self.device: Annotated[Union[str, None], 'The device to use for computation'] = None
        # Evaluating args
        # IR setting

    def from_dict(self, args_dict):
        for key in args_dict:
            if key in self.__dict__:
                setattr(self, key, args_dict[key])

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"