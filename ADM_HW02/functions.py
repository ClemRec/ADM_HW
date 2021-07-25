'''
This python file strores all functions our group used for the ADM-Homework 2.
It stores codes by questions even if not all questions request for an implemented function.
'''

#import part
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import operator
import re # This one is used in RQ2 and RQ3
import statistics # This one is used in RQ3


'''
RQ1
'''

def RQ1_0(ds):
    # selecting what we are interested in
    v= ds[ds.event_type=="view"].groupby([ds.user_id,ds.product_id])
    c= ds[ds.event_type=="cart"].groupby([ds.user_id,ds.product_id])
    p= ds[ds.event_type=="purchase"].groupby([ds.user_id,ds.product_id])

    # computing the rate of the funnels
    TOT=0
    TOT_CFUN = 0
    l=[]
    for index,row in p:
        u,pr=index[0],index[1]

        if u not in l:
            for index_,row in c:
                if (index_[0],index_[1])==(u,pr):
                    for index__,row in v:
                        if (index__[0],index__[1])==(u,pr):
                            TOT_CFUN = TOT_CFUN + 1
                            TOT+=1
                            l.append(u)
                        else:
                            TOT+=1
                else:
                    TOT+=1
    return TOT,TOT_CFUN


def RQ1_1a(ds):
    
    # We group by user session and choose the event type we want to calculate the average on 
    AVG_VIEW = ds[ds.event_type == 'view'].groupby(ds.user_session).event_type.count().mean()
    AVG_PURCH = ds[ds.event_type == 'purchase'].groupby(ds.user_session).event_type.count().mean()
    AVG_CART = ds[ds.event_type == 'cart'].groupby(ds.user_session).event_type.count().mean()
    AVG_RMV = ds[ds.event_type == 'remove_from_cart'].groupby(ds.user_session).event_type.count().mean()
    
    # saving the results in a dictionary
    D = {'view' : AVG_VIEW, 'purchase' : AVG_PURCH, 'Remove_from_cart' : AVG_RMV, 'cart' : AVG_CART}
    
    labels = D.keys()
    values = D.values()

    if 'Remove_from_cart' in D :
        D['Remove_from_cart'] = 0

    if D['Remove_from_cart'] == 0 :
        del D['Remove_from_cart']

    #plotting
    plt.pie(values, labels=labels)

def RQ1_1b(ds):
    # We proceed with the same reasoning as the function above but this time we group by user id
    AVG_VIEW = ds[ds.event_type == 'view'].groupby(ds.user_id).event_type.count().mean()
    AVG_PURCH = ds[ds.event_type == 'purchase'].groupby(ds.user_id).event_type.count().mean()
    AVG_CART = ds[ds.event_type == 'cart'].groupby(ds.user_id).event_type.count().mean()
    AVG_RMV = ds[ds.event_type == 'remove_from_cart'].groupby(ds.user_id).event_type.count().mean()

    # We sort the averages first
    my_labels = np.array(['view', 'purchase', 'cart', 'remove_from_cart'])
    my_values = np.array([AVG_VIEW, AVG_PURCH, AVG_CART, AVG_RMV])
    indexes = np.argsort(my_values)

    sorted_values = my_values[indexes]
    sorted_labels = my_labels[indexes]

    # And then we plot them
    plt.barh(sorted_labels, sorted_values, align = 'center', color=['r', 'b', 'g', 'y'])
    plt.title ('Average number of operations per user', size=15)
    plt.show()

def RQ1_2(a):
    TOT_VIEWS = 0
    TOT_CASES = 0
    for _,row in a:
        list_events = list(row.values)   # We put the event types inside a list
        i = 0
        if 'view' in row.values:
            if 'cart' in row.values:
                cart_index = list_events.index('cart')
                if list_events.index('view') < cart_index:  # we check if the view comes before the cart 
                    TOT_CASES = TOT_CASES + 1     
                    TOT_VIEWS = TOT_VIEWS + cart_index  
    return (TOT_VIEWS / TOT_CASES)

def RQ1_3(a):
    FAV_CASES = 0
    for _,row in a:
        if 'cart' in row.values:
            if 'purchase' in row.values:
                arr = np.array(row.values)
                Ncart = np.in1d(arr, 'cart').sum()     # We count how many cart events there are 
                Npurchase = np.in1d (arr, 'purchase').sum()    # We count how many purchase events there are
                FAV_CASES = FAV_CASES + min(Ncart, Npurchase)  
    return FAV_CASES

def RQ1_4(a):
    TOT_DATES = datetime.timedelta(0)
    TOT_CASES = 0
    for _, row in a:
        if 'cart' in row.values:
            if 'purchase' in row.values:
                
                # We find the date when some user used a cart type
                date_cart = row.loc[row.event_type == 'cart'].event_time.iloc[0] 
                
                # We find the date when some user used a purchase type
                date_purch = row.loc[row.event_type == 'purchase'].event_time.iloc[0] 

                # We compute the differences between these two dates
                date_diff = date_purch - date_cart  

                # We check if the cart event came before the purchase event(e.g. [purchase, cart]), and in that case we skip it  
                
                if date_diff > datetime.timedelta(0):   
                    TOT_CASES = TOT_CASES+1
                    TOT_DATES = TOT_DATES + date_diff
    return TOT_DATES,TOT_CASES


def RQ1_5(a):
    TOT_DATES = datetime.timedelta(0)
    TOT_CASES = 0
    for _, row in a:
        if 'view' in row.event_type.values:

                    date_view = row.loc[row.event_type == 'view'].event_time.iloc[0]
                    date_diff = datetime.timedelta(-1)
                    i = 0

                    while date_diff < datetime.timedelta(0) : 

                            
                          
                        
                        # We check if there is a cart event and we take the difference between the date of view and the date of cart
                        if 'cart' in row[row.event_type == 'cart'].event_type.values[i:] :
                            date_cart = row.loc[row.event_type == 'cart'].event_time.iloc[i]
                            date_diff1 = date_cart - date_view

                        else :
                            date_diff1 = datetime.timedelta(0)


                        # We check if there is a purchase event and we take the difference between the date of view and the date of purchase
                        if 'purchase' in row[row.event_type == 'purchase'].event_type.values[i:]:
                            date_purch = row.loc[row.event_type == 'purchase'].event_time.iloc[i]
                            date_diff2 = date_purch - date_view

                        else :
                            date_diff2 = datetime.timedelta(0)

                        i = i + 1

                        # if both the differences are positive (i.e. the view comes before cart and purchase) we just take the minimum of 
                        # these two differences because we want the one that came first between the two events 
                        if date_diff1 > datetime.timedelta(0) and date_diff2 > datetime.timedelta(0):
                            date_diff = min(date_diff1, date_diff2)

                        else :
                            # we only have a cart event (without a purchase) after a view 
                            if date_diff1 > datetime.timedelta(0):
                                date_diff = date_diff1
                            
                            # we only have a purchase event (without a cart) after a view
                            elif date_diff2 > datetime.timedelta(0):
                                date_diff = date_diff2
                            
                            # if we only have a cart event before a view, we reiterate the while cycle to get the next cart event
                            elif date_diff1 < datetime.timedelta(0):
                                date_diff = date_diff1
                            
                             # if we only have a purchase event before a view, we reiterate the while cycle to get the next cart event
                            elif date_diff2 < datetime.timedelta(0):
                                date_diff = date_diff2
                            
                            # if we don't have purchase or cart events, we skip this row 
                            elif date_diff1 == datetime.timedelta(0) and date_diff2 == datetime.timedelta(0):
                                date_diff = datetime.timedelta(0)
                                TOT_CASES = TOT_CASES - 1

                    TOT_CASES = 1 + TOT_CASES
                    TOT_DATES = TOT_DATES + date_diff

    return TOT_DATES,TOT_CASES





'''
RQ2
'''

'Q1 : For each month visualize this information through a plot showing the number of sold products per category.'

# Here we construct the dictionary containing the primary category (means the category before the first point) based on products in the giving groupbyobject 'a' in parameters.

def preproc(a):
    t=[]
    for index,row in a:
        if re.search(r'[a-zA-Z0-9\_]{1,}\.([a-zA-Z0-9\_]\.?){1,}',str(index)): #regex request to separate the primary category to the sub-categories
            t.append(re.search(r'[a-zA-Z0-9\_]{1,}\.',str(index)).group(0))    #retrieve just the primary category from the category_code
    t=list(set(t))
    dic={}
    for i in range(len(t)): #building a dictionnary that assumes primary categories as keys
        dic[t[i]]=0
    return dic



# Here we create a function that returns the max number of sales for a given groupbyobject, the category that corresponds to and a dataframe build from the dictionary built in the previous code with "preproc()" function.

def RQ2_1(a):
    dic=preproc(a)
    for index,row in a:
        test=re.search(r'[a-zA-Z0-9\_]{1,}\.',str(index)).group(0)
        dic[test]+=row.product_id.count()   #count the number of sales and add it in the dictionnary according to the good primary category
    maxi=0
    cat=''
    for i in dic.keys():  #max and cat are piles and we compare for each category it number of sales and retrieve the category with the max sells and it name
        if dic[i]>maxi:
            maxi=dic[i]
            cat=cat.replace(cat,i)
        elif dic[i]==maxi:
            cat+= ' '+ i  #here it is a "+" to take draw into account, for instance if two categories got the same number of sales
        df=pd.DataFrame(dic, index=[0])
    return df,cat,maxi


'Q2 : Plot the most visited subcategories'

# First of all create a function looking for all subcategories of the requested primary category.
def search_sub(a,cat_code):
    tab=[]
    dic={}
    for index,row in a:
        if cat_code in re.search(r'[a-zA-Z0-9\_]{1,}\.',str(index)).group(0):
            tab.append(index)
    if len(tab)==0:
        print("I am sorry you may have make a mistake writing the name of the category")
    for i in range(len(tab)):
        temp=re.search(r'[a-zA-Z0-9\_]{1,}\.([a-zA-Z0-9\_]{1,}\.?){1}',tab[i])  #Regex request to retrieve the subcategory name.
        if temp:
            if temp.group(0) not in dic.keys():
                dic[temp.group(0)]=0
            else:
                dic[temp.group(0)]+=0
    return dic #return a dictionnary containing the name of subcategories as keys


#We create a function which searches in the groupbyobject for existing sub-categories of the primary category given in parameters.
#Then we group them in a dictionary and count for each subcategories the number of views
def fctn_(a,cat_code):
    dic=search_sub(a,cat_code)
    for index,row in a:
        test=re.search(r'[a-zA-Z0-9\_]{1,}\.([a-zA-Z0-9\_]{1,}\.?){1}',str(index)).group(0)
        if test in dic.keys():
            dic[test]+=row.product_id.count()
    print(dic)
    maxi=0
    subcat=''
    for i in dic.keys():
        if dic[i]>maxi:
            maxi=dic[i]
            subcat=i
        elif dic[i]==maxi:
            subcat+= ' '+ i
        df=pd.DataFrame(dic, index=[0])
    return df,subcat,maxi


# In this function we call previous functions to plot and print what is asking in this question.
def RQ2_2(a):
    dic=preproc(a)
    liste=list(dic.keys())
    for i in range(len(liste)):
        df,subcat,maxi=fctn_(a,liste[i])
        print(df.plot.bar(figsize=(12,3), title='Number of views of sub categories of the {} category'.format(liste[i])))
        res='\033[1m'+'The sub category of {} which has the maximum views is {} with {} views'.format(liste[i], subcat, maxi)+'\033[0m'
        res=res.replace('.',' ')
        print(res)



'Q3 : What are the 10 most sold products per category?'

# Here we first create a function that for a given primary category, returns the 10 most sold products per category.
def nb_max_sales(a,primary):
    tab=[]
    for index, row in a:
        temp=re.search(r'[a-zA-Z0-9\_]{1,}\.',str(index[0])).group(0)
        if primary in temp:
            tab.append((row.product_id.count(),index[1])) #create a list and add a tuple containing the number of sales and the corresponding product_id
        tab.sort(reverse=True) #sort it in descending order
    uless=tab[:9]              #keep 10 first element
    nb_sells=[]
    product_id=[]
    for i in range(len(uless)):
        nb_sells.append(uless[i][0])
        product_id.append(uless[i][1])
    return nb_sells,product_id


def RQ2_3(a):
    dico=preproc(a)
    for i in dico.keys():
        nb_sells,product_id=nb_max_sales(a,i) #as we already done in the previous question, we using "proproc()" to loop on all primary categories name and apply "nb_max_sales()"
        string=''
        for j in product_id:
            string+=' '+str(j)
        res='\033[1m'+'The 10 most sold products of the {} category are {}.'.format(i, string)+'\033[0m'
        res=res.replace('.','',1)
        print(res)


'''
RQ3
'''

'Q1 : Write a function that asks the user a category in input and returns a plot indicating the average price of the products sold by the brand.'

def RQ3_1(a):
    print('Insert a category: ')
    primary=input() #ask for the user which category he looking for
    dico={}
    for index,row in a:
        temp=re.search(r'[a-zA-Z0-9\_]{1,}\.',str(index[0])).group(0)
        if primary in temp:                 # in this loop, index[0],index[1],index[2] respectively correspond to category_code, brand, price
            if index[1] not in dico.keys(): #building a dictionnary containing for each brand all prices of it sold products
                dico[index[1]]=[index[2]]
            else:
                dico[index[1]].append(index[2])
    if len(dico.keys())==0:
        print("I am sorry you may have make a mistake writing the name of the category")
    for i in dico.keys():
        dico[i]=statistics.mean(dico[i])  #calcul for each brand the mean price

    plt.xticks(rotation='vertical')
    plt.bar(dico.keys(), dico.values(), linewidth=4)
    plt.show()

    a=(max(dico.values())) #looking for the max average over all brands
    for i in dico.keys():
        if dico[i]==a:
            print('\033[1m'+'The brand whose prices are higher on average in the {} category is {}.'.format(primary,i)+'\033[0m')



def RQ3_2(a,primary):
    dico={}
    for index,row in a:
        temp=re.search(r'[a-zA-Z0-9\_]{1,}\.',str(index[0])).group(0)
        if primary in temp:
            if index[1] not in dico.keys():
                dico[index[1]]=[index[2]]
            else:
                dico[index[1]].append(index[2])

    for i in dico.keys():
        dico[i]=statistics.mean(dico[i])

    l=sorted(dico.items(), key=lambda t: t[1])   # order by means

    print('\033[1m'+"The brand whose prices are higher on average in the {} category is {} with an average product price around {}.".format(primary, l[-1][0], round(l[-1][1]))+'\033[0m')
    for i in range(len(l)-1):
        print("Then it comes "+str(l[i][0])+" with an average of "+ str(round(l[i][1]))+"euros.")


def print_RQ3_2(y): #looping on all categories name thanks to "preproc()"
    dic=preproc(y)
    for i in dic.keys():
        _=RQ3_2(y,i)



'''
RQ4
'''


def RQ4_1 (brand,df) :
    months = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
    L = []
    if brand in df.brand[df.brand.notna()].unique(): # We check if the input brand is in our dataframe 
        x = df[(df.event_type == 'purchase') & (df.brand == brand)].groupby(df.event_time.dt.month).price.sum()
        for i in x.index:
            L.append([months[i-1], round(x.at[i])])
        return(L)
    else :
        print ('Error: The brand you inserted does not exist')

# Notice that altough that dataset we use here is composed by just two months the function above is general and applicable
# to a dataset composed of 12 months


# First we want to find for each brand it's worst loss in the year
def RQ4_2(df):
    G = []
    for brand in df.brand[df.brand.notna()].unique():
        L = RQ4_1(brand,df)
        for i in range(1,len(L)):
            if L[i-1][1] - L[i][1] > 0:     # We check if between the two months there was a loss
                perc = ( (L[i-1][1] - L[i][1]) / L[i-1][1] ) * 100   # We apply the formula for the conversion rate for each brand 
                G.append([brand, L[i-1][0],L[i][0], perc]) # We create a list of elements: [brand, Old month, New month, conversion rate] 
    sorted_G = sorted(G, key=operator.itemgetter(3), reverse=True) # We sort the list by their conversion rate
    for x,y,z,w in sorted_G[0:3]:
        print (x, end = ' ')
        print ('lost %d%%' %w, end=' ')
        print('between ' + y + ' and ' + z)


'''
RQ5
'''



'''
RQ6
'''


def RQ6_2a(a,df):
    tmp = a.index.tolist()
    D = {}
    for i in range(len(tmp)):
        tmp[i] = tmp[i].split('.')[0]
    categories = list(set(tmp))

    for x in categories:
        sub_categories = [y for y in a.index.tolist() if x in y.split('.')[0]]
        D[x] = sub_categories      # We create a dictionary of elements: {'category' : subcategories}

    Q = {}  #Then we create a dictionary that has the category as index and the number of purchases as value
    TOT_PRICES = 0
    for category in D:
        TOT_PRICES = 0
        for sub_category in D[category]:
                TOT_PRICES = TOT_PRICES + df[(df.event_type == 'purchase') & (df.category_code == sub_category)].event_type.count()
        Q[category] = TOT_PRICES  
    return Q,D


def RQ6_2b(a,df):
    Q,D=RQ6_2a(a,df)
    G = []
    for category in D:
        TOT_VIEWS = 0
        for sub_category in D[category]:
                TOT_VIEWS = TOT_VIEWS + df[(df.event_type == 'view') & (df.category_code == sub_category)].event_type.count()

        CONV_RATE = np.divide( Q[category], TOT_VIEWS )*100
        G.append([category, CONV_RATE])

    sorted_G = sorted(G, key=operator.itemgetter(1), reverse=True)
    for x,y in sorted_G :
        print ('\033[1m' + x +'\033[0m', end = '        ')
        print(y)



'''
RQ7
'''

def Pareto_purchases(ds):
    ds_purchase = ds[ds.event_type == 'purchase'] #sub-dataset

    # counting the purchasers and the amount of significant purchasers
    tot_number_of_purchaser = ds_purchase.user_id.nunique()

    number_significant_purchasers = int(0.2*tot_number_of_purchaser)

    # compute the profit
    total_profit = ds_purchase.groupby("user_id").agg({'price':sum}).sum().round()

    profit_significant_purchasers = ds_purchase.groupby("user_id").agg({'price':sum})\
    .price.nlargest(number_significant_purchasers).sum().round()

    # Does Pareto apply?
    # because getting exactly the 80% does not seem realistic, we evaluete if Pareto applies in an interval [0.795 , 0.805]
    if (profit_significant_purchasers >= 0.79 * total_profit).any() & (profit_significant_purchasers <= 0.81 * total_profit).any():
        print("The Pareto principle does apply to our store")
    else:
        print("The Pareto principle does NOT apply to our store")