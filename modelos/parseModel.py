import io
net = "resnet50"
origin = f"../../jetson/gemmEvaluation/malleableDNN/{net}.csv"
destination = f"{net}.csv"
nL =0
buff = io.StringIO()

out = open(destination,'w')
with open(origin,'r') as f:
    f.readline()#skip title line
    lines = f.readlines()
    for prev, curr in zip(lines,lines[1:]):
        prevLayer = prev.split(';')
        layer = curr.split(';')
        if(layer[1] =="conv"):
            convLayer =f"{layer[0]};{';'.join(prevLayer[2:5])};{';'.join(layer[5:7])};{layer[4]};{layer[7]};0\n"
            buff.write(convLayer)
            nL+=1

out.write(f"nConv={nL}\n")
out.write("id;height;width;channels;kernel_heigth;kernel_width;kernel_num;stride;padding\n")
out.write(buff.getvalue())
out.close()
