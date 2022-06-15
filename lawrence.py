import os
import datetime
import pandas as pd

log_path = './yolo/log/0616_thres/testlog'
log_files = os.listdir(log_path)

time_path = './yolo/log/0616_thres/time'
time_files = os.listdir(time_path)

SAMPLE_TIME = 3

time_df = pd.read_csv(os.path.join(time_path, time_files[0]))

thres02_time_list = time_df['0.2 threshold'].to_list()
thres07_time_list = time_df['0.7 threshold'].to_list()



columns02 = ['sample_num', 'detect 0.2', 'total 0.2']
columns07 = ['sample_num', 'detect 0.7', 'total 0.7']

for log in log_files:

    df02 = pd.DataFrame(columns=columns02)
    df07 = pd.DataFrame(columns=columns07)

    with open(os.path.join(log_path, log), 'r') as f:
        lines = f.read()

        lines = lines.split('\n')

        if lines[-1] == '':
            del lines[-1]

        detect02 = 0
        total02 = 0
        sample_num = 1

        for t02 in thres02_time_list:
            start_time = datetime.datetime.strptime(t02, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.datetime.strptime(str(start_time + datetime.timedelta(seconds=SAMPLE_TIME)),
                                                  '%Y-%m-%d %H:%M:%S')

            for line in lines:
                now = datetime.datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')

                if start_time <= now < end_time:
                    total02 += 1
                    is_detect = line[27:]

                    if is_detect == 'Detect':
                        detect02 += 1

                elif now >= end_time:
                    # add the sample data to dataframe
                    data = [(sample_num, detect02, total02)]
                    df02 = pd.concat([df02, pd.DataFrame(data, columns=columns02)], ignore_index=True)

                    # initialize
                    total02 = 0
                    detect02 = 0
                    sample_num += 1
                    break


        detect07 = 0
        total07 = 0
        sample_num2 = 1

        for t07 in thres07_time_list:
            start_time = datetime.datetime.strptime(t07, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.datetime.strptime(str(start_time + datetime.timedelta(seconds=SAMPLE_TIME)),
                                                  '%Y-%m-%d %H:%M:%S')

            for line in lines:
                now = datetime.datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')

                if start_time <= now < end_time:
                    total07 += 1
                    is_detect = line[27:]

                    if is_detect == 'Detect':
                        detect07 += 1

                elif now >= end_time:
                    # add the sample data to dataframe
                    data = [(sample_num2, detect07, total07)]
                    df07 = pd.concat([df07, pd.DataFrame(data, columns=columns07)], ignore_index=True)

                    # initialize
                    total07 = 0
                    detect07 = 0
                    sample_num2 += 1
                    break


    df02.to_excel('./data/{}02_data.xlsx'.format(log[:-4]), index=False)
    df07.to_excel('./data/{}07_data.xlsx'.format(log[:-4]), index=False)
