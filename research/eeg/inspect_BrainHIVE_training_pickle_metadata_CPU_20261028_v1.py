"""Symbolically inspect a known training pickle while seeking past EEG bytes.

No pickle GLOBAL/REDUCE/BUILD callable is executed. Large binary payloads are
represented by length/offset records. Only small named time/channel metadata
are decoded, never the EEG ndarray. Unsupported opcodes fail closed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import time
import numpy as np


class Symbol:
    def __init__(self,name,args=None):self.name=name;self.args=args;self.state=None


class Skipped:
    def __init__(self,size,offset):self.size=size;self.offset=offset


class Reader:
    def __init__(self,stream,total,cap):self.stream=stream;self.total=total;self.cap=cap;self.read_bytes=0;self.skipped=[]
    def read(self,n):
        assert 0<=n<=self.cap,'Only bounded non-EEG pickle arguments are read'
        assert self.stream.tell()+n<=self.total
        raw=self.stream.read(n);assert len(raw)==n;self.read_bytes+=n;assert self.read_bytes<8<<20
        return raw
    def readline(self):
        raw=self.stream.readline(65536);assert raw.endswith(b'\n');self.read_bytes+=len(raw);assert self.read_bytes<8<<20;return raw
    def tell(self):return self.stream.tell()
    def payload(self,n):
        assert 0<=n<=self.total-self.tell()
        if n<=self.cap:return self.read(n)
        item=Skipped(n,self.tell());self.stream.seek(n,1);self.skipped.append(dict(bytes=n,offset=item.offset));return item


def parse(stream,total,cap=2<<20):
    r=Reader(stream,total,cap);stack=[];memo={};mark=object();operations=0
    def marked():
        position=len(stack)-1
        while position>=0 and stack[position] is not mark:position-=1
        assert position>=0
        result=stack[position+1:];del stack[position:];return result
    while True:
        operations+=1;assert operations<=20000
        code=r.read(1).decode('latin1');op=pickletools.code2op[code];name=op.name
        if name in ['BINBYTES','BINBYTES8','SHORT_BINBYTES','BYTEARRAY8']:
            n=int.from_bytes(r.read(1 if name=='SHORT_BINBYTES' else 4 if name=='BINBYTES' else 8),'little')
            arg=r.payload(n)
        else:arg=op.arg.reader(r) if op.arg else None
        if name in ['PROTO','FRAME']:pass
        elif name=='STOP':
            assert len(stack)==1 and r.tell()==total
            return stack[0],dict(opcodes=operations,actually_read_metadata_bytes=r.read_bytes,skipped_binary_payloads=r.skipped,GLOBAL_REDUCE_BUILD_callables_executed=0)
        elif name=='MARK':stack.append(mark)
        elif name in ['NONE','NEWTRUE','NEWFALSE']:stack.append(None if name=='NONE' else name=='NEWTRUE')
        elif name in ['BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4','FLOAT','BINFLOAT','UNICODE','BINUNICODE','BINUNICODE8','SHORT_BINUNICODE','STRING','BINSTRING','SHORT_BINSTRING','BINBYTES','BINBYTES8','SHORT_BINBYTES','BYTEARRAY8']:stack.append(arg)
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='TUPLE':stack.append(tuple(marked()))
        elif name in ['TUPLE1','TUPLE2','TUPLE3']:
            count=int(name[-1]);value=tuple(stack[-count:]);del stack[-count:];stack.append(value)
        elif name=='LIST':stack.append(marked())
        elif name=='DICT':
            values=marked();assert len(values)%2==0;stack.append(dict(zip(values[::2],values[1::2])))
        elif name=='APPEND':value=stack.pop();assert isinstance(stack[-1],list);stack[-1].append(value)
        elif name=='APPENDS':values=marked();assert isinstance(stack[-1],list);stack[-1].extend(values)
        elif name=='SETITEM':value=stack.pop();key=stack.pop();assert isinstance(stack[-1],dict);stack[-1][key]=value
        elif name=='SETITEMS':
            values=marked();assert len(values)%2==0 and isinstance(stack[-1],dict);stack[-1].update(zip(values[::2],values[1::2]))
        elif name in ['BINPUT','LONG_BINPUT','PUT']:memo[int(arg)]=stack[-1]
        elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif name in ['BINGET','LONG_BINGET','GET']:stack.append(memo[int(arg)])
        elif name=='GLOBAL':stack.append(Symbol(arg.replace(' ','.')))
        elif name=='STACK_GLOBAL':name_value=stack.pop();module=stack.pop();assert isinstance(module,str) and isinstance(name_value,str);stack.append(Symbol(module+'.'+name_value))
        elif name=='REDUCE':args=stack.pop();function=stack.pop();assert isinstance(function,Symbol) and isinstance(args,tuple);stack.append(Symbol(function.name,args))
        elif name=='BUILD':state=stack.pop();assert isinstance(stack[-1],Symbol);stack[-1].state=state
        elif name=='POP':stack.pop()
        elif name=='POP_MARK':marked()
        elif name=='DUP':stack.append(stack[-1])
        else:raise ValueError('Unsupported symbolic pickle opcode: '+name)
        assert len(stack)<20000 and len(memo)<20000


def array_parts(value):
    if not isinstance(value,Symbol):return None
    if value.name.endswith('._reconstruct'):
        version,shape,dtype,fortran,payload=value.state
        assert version==1 and isinstance(shape,tuple)
    elif value.name.endswith('._frombuffer'):
        payload,dtype,shape,order=value.args;fortran=order=='F'
    else:return None
    assert isinstance(dtype,Symbol) and dtype.name=='numpy.dtype' and isinstance(dtype.args[0],str)
    dt=dtype.args[0]
    if dtype.state is not None and isinstance(dtype.state,tuple) and dtype.state[1] in ['<','>','|','='] and dt[0] not in '<>|=':dt=dtype.state[1]+dt
    return shape,dt,fortran,payload


def summarize(value,decode=False):
    parts=array_parts(value)
    if parts is not None:
        shape,dt,fortran,payload=parts
        record=dict(kind='symbolic_numpy_array',shape=list(shape),dtype=dt,Fortran_order=bool(fortran),EEG_values_decoded=False)
        if isinstance(payload,Skipped):record.update(binary_payload_skipped_bytes=payload.size,binary_payload_offset=payload.offset)
        elif isinstance(payload,(bytes,bytearray)):
            record['small_payload_bytes']=len(payload)
            if decode:
                dtype=np.dtype(dt);assert dtype.kind in 'fiuUS' and not dtype.hasobject and len(shape)<=1 and int(np.prod(shape))<=2000
                data=np.frombuffer(payload,dtype=dtype).reshape(shape,order='F' if fortran else 'C')
                record['values']=data.tolist();record['small_named_metadata_decoded']=True
                if data.dtype.kind in 'fiu' and data.size>1:
                    step=np.diff(data);record.update(first=float(data[0]),last=float(data[-1]),minimum_step=float(step.min()),maximum_step=float(step.max()))
        elif isinstance(payload,list) and decode and len(payload)<=2000 and all(isinstance(x,(str,int,float,bool,type(None))) for x in payload):record['values']=payload
        return record
    if isinstance(value,(str,int,float,bool,type(None))):return dict(kind='literal',value=value)
    if isinstance(value,(list,tuple)) and len(value)<=2000 and all(isinstance(x,(str,int,float,bool,type(None))) for x in value):return dict(kind='literal_sequence',values=list(value))
    return dict(kind=type(value).__name__,symbolic_callable=value.name if isinstance(value,Symbol) else None,not_executed=True)


def self_test():
    original=dict(preprocessed_eeg_data=np.arange(10000,dtype=np.float32).reshape(10,10,100),times=np.arange(10,dtype=np.float64)/250,ch_names=['O1','Oz','O2'])
    checks=[]
    for protocol in (3,4,5):
        data=pickle.dumps(original,protocol=protocol);value,proof=parse(io.BytesIO(data),len(data),cap=1024)
        assert isinstance(value,dict) and list(value)==list(original)
        record=summarize(value['preprocessed_eeg_data']);assert record['shape']==[10,10,100] and record['binary_payload_skipped_bytes']==40000
        assert np.array_equal(summarize(value['times'],True)['values'],original['times']) and summarize(value['ch_names'])['values']==original['ch_names']
        assert proof['actually_read_metadata_bytes']<2048 and proof['GLOBAL_REDUCE_BUILD_callables_executed']==0
        checks.append('NumPy protocol'+str(protocol)+' exact metadata, large payload skipped')
    # This would return3 under pickle.loads. The symbolic parser retains the
    # unevaluated call instead; it never invokes a pickle-selected callable.
    data=b'cbuiltins\nlen\n(Vabc\ntR.';value,proof=parse(io.BytesIO(data),len(data))
    assert isinstance(value,Symbol) and value.name=='builtins.len';checks.append('REDUCE retained as symbolic call, not executed')
    return dict(status='SYMBOLIC_PICKLE_METADATA_FOUR_CONTRACTS_PASSED',checks=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--self-test-only',action='store_true');a=p.parse_args()
    assert not a.output.exists();start=time.monotonic();tests=self_test()
    if a.self_test_only:result=tests
    else:
        expected=Path('/root/autodl-tmp/atm_published_training_trials_20261016_v2/preprocessed/sub-01/preprocessed_eeg_training.npy');assert a.source==expected
        with expected.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest=='c54cf2e8a4f0c122d79bb7d98ddf04698f992d292f3e26ae59bed97230944f63'
        with expected.open('rb') as f:value,proof=parse(f,expected.stat().st_size)
        assert isinstance(value,dict)
        records={str(key):summarize(item,decode=str(key).lower() in ['times','time','ch_names','channels','sfreq','sampling_rate']) for key,item in value.items()}
        result=dict(status='PINNED_TRAINING_PICKLE_SYMBOLIC_METADATA_ONLY_COMPLETE',source=str(expected),source_sha256=digest,tests=tests,proof=proof,top_level_keys=list(map(str,value)),fields=records,
                    EEG_payload_unpickled_or_decoded=False,pickle_callables_executed=False,new_GPU_jobs=0,new_model_fits=0,new_images=0)
    result.update(seconds=time.monotonic()-start,source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
