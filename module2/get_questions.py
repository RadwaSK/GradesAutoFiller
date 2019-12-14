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