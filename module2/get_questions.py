import numpy as np

#Getting Questions clusters
def arrangeList(Y_cord):
    Y_cord = sorted(Y_cord)
    ret = []
    for y in Y_cord:
        ret.append(y[1])
    return ret

def getCols(bubbles):
    X_cord = []
    EPS = np.min(bubbles[:,1]-bubbles[:,0])
    for i in range(0,len(bubbles)):
        X_cord.append([bubbles[i,0],i])
    X_cord = sorted(X_cord)
    st=0
    Cols = []
    c = []
    for i in range(0,len(X_cord)):
        if X_cord[i][0]-X_cord[st][0] > EPS:
            Cols.append(arrangeList(c))
            c = []
            st=i
        ind = X_cord[i][1]
        c.append([bubbles[ind,2],ind])
    Cols.append(arrangeList(c))
    return Cols

# Check for different rows & columns lengths
def getRows(cols):
    R = []
    for j in range(0,len(cols[0])):
        R.append([])
    for i in range(0,len(cols)):
        for j in range(0,len(cols[i])):
            R[j].append(cols[i][j])
    return R

def find_questions(rows,bubbles):
    mn=10**10
    mx=-1
    questions=[]
    for i in range (0,len(rows[0])-1):
        dff=bubbles[rows[0][i+1]][0]-bubbles[rows[0][i]][1]      #xmin of nxt- my xmax
        mn=min(dff,mn)
        mx=max(dff,mx)
    if((1.0*mx)/mn<1.5):
        return rows
    n=1                     #number of questions in the largest row (1st row)
    for i in range (0,len(rows[0])-1):
        dff=bubbles[rows[0][i+1]][0]-bubbles[rows[0][i]][1]    
        if(abs(dff-mx)<0.1*mx):
            n+=1
    ch=len(rows[0])//n          #number of choices for every question
    
    for i in range (0,n):
        que=[]
        for j in range (0,len(rows)):
            if(i*ch>=len(rows[j])):
                break
            que=rows[j][i*ch:(i+1)*ch]
            questions.append(que)
    return questions