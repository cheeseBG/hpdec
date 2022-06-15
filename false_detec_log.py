import os
import datetime
import pandas as pd

log_path = './yolo/log/0616'
log_files = os.listdir(log_path)

SAMPLE_TIME = 300
START_TIME = '2022-06-15 23:35:00'
END_TIME = '2022-06-15 23:40:00'

columns = ['sample_num', 'detect', 'total', '0.2', '0.7']

for log in log_files:

    df = pd.DataFrame(columns=columns)

    with open(os.path.join(log_path, log), 'r') as f:
        lines = f.read()

        lines = lines.split('\n')

        if lines[-1] == '':
            del lines[-1]

        start_time = datetime.datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.datetime.strptime(str(start_time + datetime.timedelta(seconds=SAMPLE_TIME)), '%Y-%m-%d %H:%M:%S')

        count = 0
        detect = 0
        thres02 = 0
        thres07 = 0
        sample_num = 1

        for line in lines:
            now = datetime.datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
            if start_time >= datetime.datetime.strptime(END_TIME, '%Y-%m-%d %H:%M:%S'):
                break

            if start_time <= now < end_time:
                count += 1
                is_detect = line[27:]
                thres = line[-3:]
                if 'Detect' in is_detect:
                    detect += 1

                if thres == '0.2':
                    thres02 += 1
                elif thres == '0.7':
                    thres07 += 1

            elif now >= end_time:
                # add the sample data to dataframe
                data = [(sample_num, detect, count, thres02, thres07)]
                df = pd.concat([df, pd.DataFrame(data, columns=columns)], ignore_index=True)

                # initialize
                count = 0
                detect = 0
                sample_num += 1

                start_time = end_time
                end_time = start_time + datetime.timedelta(seconds=SAMPLE_TIME)

                if start_time <= now < end_time:
                    count += 1
                    is_detect = line[27:]
                    if 'Detect' in is_detect:
                        detect += 1
                    if thres == '0.2':
                        thres02 += 1
                    elif thres == '0.7':
                        thres07 += 1

    df.to_excel('./data/{}_data.xlsx'.format(log[:-4]), index=False)
