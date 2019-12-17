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
    percent_list = []
    for qst in range (0,len(questions)):
        for ch in range(0,choices):
            choice=questions[qst][ch]
            xmn=bubbles[choice][0]
            xmx=bubbles[choice][1]
            ymn=bubbles[choice][2]
            ymx=bubbles[choice][3]
            part=np.copy(img[ymn:ymx,xmn:xmx])
            part=cv2.threshold(part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            part=255-part
            percent= np.sum(part)/(255*(xmx-xmn)*(ymx-ymn))
            percent_list.append(percent)
    
    men_percent = np.mean(np.array(percent_list))
    vari_percent = np.var(np.array(percent_list))
    threshold = men_percent+vari_percent
    
    ch=0
    cnt=0
    for i in range(0,len(percent_list)):
        if i%choices==0:
            if cnt>1:
                answers[-1]='#'
                
            answers.append('#')
            cnt=0
            ch=0
        if(percent_list[i]>threshold):
            cnt+=1
            answers[-1]=chr(ch+65)
        ch+=1
    if cnt>1:
        answers[-1]='#'
    return answers
