## 数据位置 SMILE的data就是直接从其github上下的data
DATA_PATH = {'PsyQA_train':'/data2/AutoInvoice/large_span_enstra_PsyQA_train.json'}
CLIENT_PATH = 'http://localhost:8080'

## embedding model的位置
MODEL_PATH = '/data2/AutoInvoice/bge-large-zh-v1.5'

## 向量数据库的放置位置
PROBOOKS_DB_PATH = '/data2/AutoInvoice/ProDB-bge-1.5-300'
KNOW_DB_PATH = '/data2/AutoInvoice/KnowDB-bge-1.5-300'

## LLMs的放置位置
CHATGLM_PATH = '/data2/AutoInvoice/chatglm3-6b'

LOG_DIR = './logs'
STATIC_PATH = '/home/whh/project/Xinhai/static'

CONTROLLER_HEART_BEAT_EXPIRATION = 130
WORKER_HEART_BEAT_INTERVAL = 15

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
PREDICT_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEFAULT_PREDICT_TOKEN = "<predict>"

DESCRIPT_PROMPT = [
    "Describe this image thoroughly.",
    "Provide a detailed description in this picture.",
    "Detail every aspect of what's in this picture.",
    "Explain this image with precision and detail.",
    "Give a comprehensive description of this visual.",
    "Elaborate on the specifics within this image.",
    "Offer a detailed account of this picture's contents.",
    "Describe in detail what this image portrays.",
    "Break down this image into detailed descriptions.",
    "Provide a thorough description of the elements in this image."]

## 使用GPT的一些参数配置
class Args:
    pass
args = Args()
args.api_key = "sk-Nn5My5HMNLUMYNNBB6D43660AeE94d1789134671268f435p"
args.proxy_api_url = "http://47.252.38.253:3040/v1"
args.model_name = "gpt-4"
args.completion_number = 1
args.temperature = 0.90
args.top_p = 0.75