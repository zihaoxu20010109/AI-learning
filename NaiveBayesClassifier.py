import math
import pandas
import sys
import time

columns = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation', 'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-from-Reservation-to-Arrival', 'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-in-Weekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 'Children', 'Babies', 'Is-Repeated-Guest',
           'Previ-ous-Cancellations', 'Previous-Booking-Not-Cancelled', 'Requested-Car-Parking-Spaces', 'Total-of-Special-Requests', 'Average-Daily-Rate','reservationsStatus']
columnsold = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation', 'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-from-Reservation-to-Arrival', 'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-in-Weekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 'Children', 'Babies', 'Is-Repeated-Guest',
           'Previ-ous-Cancellations', 'Previous-Booking-Not-Cancelled', 'Requested-Car-Parking-Spaces', 'Total-of-Special-Requests', 'Average-Daily-Rate']

def transformdata(training, testing):
    n=22
    traningdata = pandas.read_csv(training, sep=",", header=None)
    testingdata = pandas.read_csv(testing, sep=",", header=None ,usecols = [i for i in range(n)])
    traningdata = traningdata.set_axis(labels=columns, axis=1)
    testingdata = testingdata.set_axis(labels=columnsold, axis=1)
    traningdatacancel = traningdata[traningdata['reservationsStatus'] == 1]
    traningdatanocancel = traningdata[traningdata['reservationsStatus'] == 0]
    return (traningdatacancel, traningdatanocancel, testingdata)



def GaussianProbability(x, mean, std):
    answer = 1.0/(std * math.sqrt(2 * math.pi))
    index = -0.5 * math.pow((x - mean)/std, 2)
    answer = answer * math.pow(math.e, index)

    if (answer < math.pow(math.e, -10)):
        answer = math.pow(math.e, -10)

    return answer


categorical = ['Hotel-Type', 'Meal-Request', 'Market-Segment-Designation',
               'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type']
categoricalIndexes = [0, 2, 3, 4, 5, 6, 7]
# remove the Requested-Car-Parking-Spaces and Reserved-Room-Type in this question.
numerical = ['Arrival-Date-Month', 'Days-from-Reservation-to-Arrival', 'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-in-Weekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 'Children', 'Babies', 'Is-Repeated-Guest',
             'Previ-ous-Cancellations', 'Previous-Booking-Not-Cancelled', 'Total-of-Special-Requests', 'Average-Daily-Rate']
numericalIndexes = [1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]


def NBclassifier(train_datasetcancel, train_datasetnocancel, Rows, my, sy, mn, sn):
    train_dataset1 = train_datasetcancel
    train_dataset2 = train_datasetnocancel
    means_cancel = my
    stds_cancel = sy

    means_nocancel = mn
    stds_nocancel = sn
    cancel = math.log(1./2, 10)
    nocancel = math.log(1./2, 10)
    for each in categorical:
        if (each == 'Hotel-Type'):
            yes = train_dataset1[each].value_counts()
            cancel += math.log(yes[Rows[0]]/35379, 10)

            no = train_dataset2[each].value_counts()
            nocancel += math.log(no[Rows[0]]/35379, 10)
        if (each == 'Meal-Request'):
            yes = train_dataset1[each].value_counts()
            cancel += math.log(yes[Rows[2]]/35379, 10)

            no = train_dataset2[each].value_counts()
            nocancel += math.log(no[Rows[2]]/35379, 10)
        if (each == 'Market-Segment-Designation'):
            yes = train_dataset1[each].value_counts()
            cancel += math.log(yes[Rows[3]]/35379, 10)

            no = train_dataset2[each].value_counts()
            nocancel += math.log(no[Rows[3]]/35379, 10)
        if (each == 'Booking-Distribution-Channel'):
            yes = train_dataset1[each].value_counts()
            cancel += math.log(yes[Rows[4]]/35379, 10)

            no = train_dataset2[each].value_counts()
            nocancel += math.log(no[Rows[4]]/35379, 10)
        if (each == 'Deposit-Type'):
            yes = train_dataset1[each].value_counts()
            cancel += math.log(yes[Rows[6]]/35379, 10)

            no = train_dataset2[each].value_counts()
            nocancel += math.log(no[Rows[6]]/35379, 10)
        if (each == 'Customer-Type'):
            yes = train_dataset1[each].value_counts()
            cancel += math.log(yes[Rows[7]]/35379, 10)

            no = train_dataset2[each].value_counts()
            nocancel += math.log(no[Rows[7]]/35379, 10)
    i = 0
    for each in numerical:
        prob_yes = GaussianProbability(
            Rows[numericalIndexes[i]], means_cancel[each], stds_cancel[each])
        cancel += math.log(prob_yes, 10)
        prob_no = GaussianProbability(
            Rows[numericalIndexes[i]], means_nocancel[each], stds_nocancel[each])
        nocancel += math.log(prob_no, 10)
        i += 1

    if (cancel > nocancel):
        return 1
    else:
        return 0


def normalize(train_dataset_cancel, train_dataset_nocancel):
    means_cancel = train_dataset_cancel.mean(axis=0, numeric_only=True)
    stds_cancel = train_dataset_cancel.std(axis=0, numeric_only=True)
    means_nocancel = train_dataset_nocancel.mean(axis=0, numeric_only=True)
    stds_nocancel = train_dataset_nocancel.std(axis=0, numeric_only=True)
    return (means_cancel, stds_cancel, means_nocancel, stds_nocancel)


def printoutput(test_dataset):
    #correct = 0
    #totalrows = 0
    #expected = test_dataset['reservationsStatus'].values.tolist()
    for row in test_dataset.values.tolist():
        prediction = NBclassifier(train_dataset_cancel, train_dataset_nocancel,
                                  row, means_cancel, stds_cancel, means_nocancel, stds_nocancel)
        print(prediction)
    """
        if prediction == 1 and expected[totalrows] == 1:
            correct += 1
        if prediction == 0 and expected[totalrows] == 0:
            correct += 1
        totalrows += 1
    return (correct, totalrows)
    """
"""
def printanswer(correct, totalrows, start, end):
    print('Accuracy: ', correct/totalrows)
    print("The time of execution of above program is :",
          (end-start) * 10**3, "ms")
"""

if __name__ == "__main__":
    start = time.time()
    datasets = transformdata(sys.argv[1], sys.argv[2])
    train_dataset_cancel = datasets[0]
    train_dataset_nocancel = datasets[1]
    test_dataset = datasets[2]
    normalSet = normalize(train_dataset_cancel, train_dataset_nocancel)
    means_cancel, stds_cancel, means_nocancel, stds_nocancel = normalSet[
        0], normalSet[1], normalSet[2], normalSet[3]
    printoutput(test_dataset)
    #correct = a[0]
    #totalrows = a[1]

    end = time.time()
    #printanswer(correct, totalrows, start, end)
