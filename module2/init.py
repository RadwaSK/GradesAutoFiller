#Reading Images
from skimage.color import rgb2gray
def appendImage(imgPath,imList):
    #Image is appended in GrayScale
    imList.append(rgb2gray(cv2.imread(imgPath,0)))
    return


def Correct(IMG,model_ans):
    Bubbles = getChoices(IMG)
    cols_list = getCols(Bubbles)
    rows_list = getRows(cols_list)
    ques = find_questions(rows_list,Bubbles)
    #################
    if len(ques) != len(model_ans):
        return None
    #####################
    ans=get_answers(ques,Bubbles,IMG)
    final_score = compareResult(ans,model_ans)
    return final_score
