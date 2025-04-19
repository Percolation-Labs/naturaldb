from rocksdict import Options, SliceTransform, Rdict
from pathlib import Path

namespace ='p8'
entity = 'Agent'
DB_PATH = str(Path.home() / '.percolate' / 'natural-db')


def main():
    opts = Options(raw_mode=True)  
    # assume all our keys share the first N bytes as the prefix:
    prefix = f"{namespace}:{entity}:vindex:".encode("utf-8")  
    opts.set_prefix_extractor(SliceTransform.create_fixed_prefix(len(prefix)))  # :contentReference[oaicite:0]{index=0}
    db = Rdict(f"{DB_PATH}/data", options=opts)       
    # ro = opts.read_options                        
    # ro.prefix_same_as_start = True
    
    it = db._db.iterator(prefix=prefix, read_options=ro)  # true C‐level prefix scan :contentReference[oaicite:3]{index=3}
    
    prefix = f"{namespace}:{entity}:vindex:".encode("utf-8")
    # (opts, db, and ro set up as above)…

    vectors, keys = [], []
    for k, v in db._db.iterator(prefix=prefix, read_options=opts):
        print(k)
        
if __name__ == '__main__':
    main()