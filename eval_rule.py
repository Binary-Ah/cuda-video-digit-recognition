#!/usr/bin/env python3
"""
파일명 규칙으로 라벨링: '(1)' 포함 -> 6, 그 외 -> 8
stretch 적용 후 mnistCUDNN 인식, 정확도 채점.

사용법:
  python eval_rule.py /home/user/Downloads/out_pgms /home/user/Downloads/mnistCUDNN/mnistCUDNN
"""
import sys, os, subprocess, re
import numpy as np

def read_pgm(p):
    with open(p,"rb") as f:
        f.readline(); w,h=map(int,f.readline().split()); f.readline()
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(h,w)

def write_pgm(p,a):
    h,w=a.shape
    with open(p,"wb") as f:
        f.write(b"P5\n"); f.write(f"{w} {h}\n".encode()); f.write(b"255\n")
        f.write(np.clip(a,0,255).astype(np.uint8).tobytes())

def stretch(a):
    a=a.astype(np.float32); mn,mx=a.min(),a.max()
    if mx>mn: a=(a-mn)/(mx-mn)*255.0
    return np.clip(a,0,255).astype(np.uint8)

def label_of(fn):
    return 6 if "(1)" in fn else 8

def classify(binary,p):
    wd=os.path.dirname(os.path.abspath(binary))
    out=subprocess.run([os.path.abspath(binary),f"image={os.path.abspath(p)}"],
                       capture_output=True,text=True,cwd=wd).stdout
    m=re.search(r"Result of classification:\s*(\d+)",out)
    t=re.search(r"Total execution time\s*:\s*([\d.]+)\s*ms",out)
    return (int(m.group(1)) if m else -1),(float(t.group(1)) if t else -1.0)

def main():
    in_dir,binary=sys.argv[1],sys.argv[2]
    out_dir=os.path.join(in_dir,"stretched"); os.makedirs(out_dir,exist_ok=True)
    files=sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".pgm"))

    res=[]
    for i,fn in enumerate(files):
        lab=label_of(fn)
        dst=os.path.join(out_dir,fn)
        write_pgm(dst,stretch(read_pgm(os.path.join(in_dir,fn))))
        pred,ms=classify(binary,dst)
        ok=(pred==lab)
        res.append((fn,lab,pred,ok,ms))
        print(f"[{i:3d}] {'O' if ok else 'X'} label={lab} pred={pred} {ms:7.1f}ms {fn}")

    six=[r for r in res if r[1]==6]; eight=[r for r in res if r[1]==8]
    s_ok=sum(r[3] for r in six); e_ok=sum(r[3] for r in eight)
    tot_ok=sum(r[3] for r in res)
    times=[r[4] for r in res if r[4]>0]
    print("\n========== 요약 (규칙: (1)=6, 그외=8) ==========")
    print(f"6 개수={len(six)}, 8 개수={len(eight)}")
    print(f"전체 정확도 : {tot_ok}/{len(res)} = {100*tot_ok/len(res):.1f}%")
    if six:   print(f"  6 정확도 : {s_ok}/{len(six)} = {100*s_ok/len(six):.1f}%")
    if eight: print(f"  8 정확도 : {e_ok}/{len(eight)} = {100*e_ok/len(eight):.1f}%")
    print(f"평균 수행시간: {np.mean(times):.1f} ms")

if __name__=="__main__":
    main()
