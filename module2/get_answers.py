import numpy as np
import cv2

#Get the answerss and the results
def compareResult(stdAns,modAns):
    score = []
    for i in range(0,len(stdAns)):
        score.append(0)
        if(stdAns[i]==modAns[i]):
            score[-1]=1
    return score

def get_answers(questions,bubbles,img):
    choices=len(questions[0])
    answers=[]
    for qst in range (0,len(questions)):
        answers.append('#')
        for ch in range(0,choices):
            cnt=0
            choice=questions[qst][ch]
            xmn=bubbles[choice][0]
            xmx=bubbles[choice][1]
            ymn=bubbles[choice][2]
            ymx=bubbles[choice][3]
            part=np.copy(img[ymn:ymx,xmn:xmx])
            part=cv2.threshold(part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            part=255-part
            percent= np.sum(part)/(255*(xmx-xmn)*(ymx-ymn))
            if(percent>0.4):
                cnt+=1
                answers[-1]=chr(ch+65)
        if(cnt>1):
            answers[-1]='#'
    return answers
