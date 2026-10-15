import scipy.io as sio, numpy as np, io, sys
from scipy.io.matlab._mio5 import MatFile5Reader
def subsystem_vars(path):
    m=sio.loadmat(path)
    b=m['__function_workspace__'].tobytes()
    hdr=b'MATLAB 5.0 MAT-file'.ljust(116,b' ')+b'\x00'*8
    r=MatFile5Reader(io.BytesIO(hdr+b[:4]+b[8:]), struct_as_record=False, squeeze_me=False)
    r.mat_stream.seek(0); r.initialize_read()
    r.mat_stream.seek(128)
    out=[]
    while True:
        try:
            h,nxt=r.read_var_header()
        except Exception as e:
            break
        try:
            out.append(r.read_var_array(h, process=True))
        except Exception as e:
            out.append(('ERR',str(e)))
        r.mat_stream.seek(nxt)
    return out
if __name__=='__main__':
    vs=subsystem_vars(sys.argv[1])
    for i,v in enumerate(vs):
        print(i,type(v),getattr(v,'dtype',None),getattr(v,'shape',None))
