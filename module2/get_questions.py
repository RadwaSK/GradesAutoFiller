#Getting Questions clusters
def arrangeList(Y_cord):
    Y_cord = sorted(Y_cord)
    ret = []
    for y in Y_cord:
        ret.append(y[1])
    return ret

def getCols():
    X_cord = []
    for i in range(0,Circles.shape[0]):
        X_cord.append([Circles[i,0],i])
    X_cord = sorted(X_cord)
    st=0
    #EPS need to be calculated somehow
    EPS = 20
    Cols = []
    c = []
    for i in range(0,len(X_cord)):
        if X_cord[i][0]-X_cord[st][0] > EPS:
            Cols.append(arrangeList(c))
            c = []
            st=i
        ind = X_cord[i][1]
        c.append([Circles[ind,1],ind])
    Cols.append(arrangeList(c))
    return Cols
def getRows(cols):
    R = []
    for j in range(0,len(cols[0])):
        new_img = np.zeros(imagesList[3].shape)
        tmpRow = []
        for i in range(0,len(cols)):
            tmpRow.append(cols[i][j])
        R.append(tmpRow)
    return R

Rows = getRows(getCols())

def find_questions(rows,Circles):
    mn=10**10
    mx=-1
    questions=[]
    for i in range (0,len(rows[0])-1):
        dff=Circles[rows[0][i+1]][0]-Circles[rows[0][i]][0]
        mn=min(dff,mn)
        mx=max(dff,mx)
    if((1.0*mx)/mn<1.5):
        return rows
    n=1
    for i in range (0,len(rows[0])-1):
        dff=Circles[rows[0][i+1]][0]-Circles[rows[0][i]][0]
        if(abs(dff-mx)<0.1*mx):
            n+=1
    ch=len(rows[0])//n
    
    for i in range (0,n):
        que=[]
        for j in range (0,len(rows)):
            if(i*ch>=len(rows[j])):
                break
            que=rows[j][i*ch:(i+1)*ch]
            questions.append(que)
    return questions


#gettings student's answers:
def get_answers(questions,circles,img):
    choices=len(questions[0])
    answers=[]
    for qst in range (0,len(questions)):
        answers.append('#')
        for ch in range(0,choices):
            cnt=0
            choice=questions[qst][ch]
            xmn=circles[choice][0]-circles[choice][2]
            xmx=circles[choice][0]+circles[choice][2]
            ymn=circles[choice][1]-circles[choice][2]
            ymx=circles[choice][1]+circles[choice][2]
            part=np.copy(img[ymn:ymx,xmn:xmx])
            part=cv2.threshold(part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            part=255-part
            percent= np.sum(part)/(255*(xmx-xmn)*(ymx-ymn))
            if(percent>0.4):
                cnt+=1
                answers[-1]=chr(ch+65)
        #if there is more than one choice
        if(cnt>1):
            answers[-1]='#'    
    return answers
#getting student's score:
def compareResult(stdAns,modAns):
    score = []
    for i in range(0,len(stdAns)):
        score.append(0)
        if(stdAns[i]==modAns[i]):
            score[-1]=1
    return score
