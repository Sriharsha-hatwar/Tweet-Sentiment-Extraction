# Write the validation functions over here.. error functions.
def download_vocab_files_for_tokenizer(tokenizer, model_type, output_path):
    '''
    This is used to download some of the vocab files and merges file for tokenizers.
    '''
    vocab_files_map = tokenizer.pretrained_vocab_files_map
    vocab_files = {}
    for resource in vocab_files_map.keys():
        download_location = vocab_files_map[resource][model_type]
        f_path = os.path.join(output_path, os.path.basename(download_location))
        urllib.request.urlretrieve(download_location, f_path)
        vocab_files[resource] = f_path
    return vocab_files

def download_files():
    # Change the name and path.
    model_type = 'roberta-base'
    output_path = "..\input\/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    vocab_files = download_vocab_files_for_tokenizer(tokenizer, model_type, output_path)
    fast_tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'))
