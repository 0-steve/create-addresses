import pandas as pd
from faker import Faker
import usaddress
from collections import defaultdict, OrderedDict
import random
import numpy as np
from datetime import date, timedelta

def create_addresses(n):
    fake = Faker()
    Faker.seed(4321)
    address_list = [fake.address() for _ in range(n)]
    return address_list

# make into function: n = number of addresses // creates address and parses label into dictionary
def parse_address(n):
    address_dict = defaultdict(list)

    fake_addresses = create_addresses(n)

    # create dict of addresses and parsed sections
    for address in fake_addresses:
        parsed_address = usaddress.parse(address)
        for parsed in parsed_address:
            address_dict[address].append(parsed)

    return address_dict

#  calls parsed addresses to remove new lines and commas
def clean_labels(n):
    address_dict = parse_address(n)
    for address in address_dict:
        for i, label in enumerate(address_dict[address]):
            label_lst = list(label)
            label_lst[0] = label_lst[0].rstrip().rstrip(',')
            label_tuple = tuple(label_lst)
            address_dict[address][i] = label_tuple

    return address_dict

# make into function: creates dictionary of addresses with labels as values + combines consecutive labels
def combineConsecutive_labels(n):
    address_dict = clean_labels(n)
    for address in address_dict:
        address_labels = address_dict[address]
        for i, label in enumerate(address_labels):
            if address_labels[i][1] == address_labels[i-1][1]:
                new_lst =  address_labels[:i-1] + address_labels[i+1:]
                new_lst.append((address_labels[i-1][0] + ' ' + address_labels[i][0], label[1]))
                address_dict[address] = new_lst
    return address_dict

# make into function:  adds missing keys to dictionary + re-orders tuple elements
def fillEmpty_keys(n):
    address_labels = {'AddressNumber', 'StreetName', 'StreetNamePostType', 'OccupancyType',
                    'OccupancyIdentifier', 'PlaceName', 'StateName', 'ZipCode', 'StreetNamePostDirectional', 
                    'StreetNamePreDirectional', 'SubaddressType', 'SubaddressIdentifier', 'USPSBoxType', 
                    'USPSBoxID', 'Recipient', 'LandmarkName'}

    address_dict = combineConsecutive_labels(n)

    for i, address in enumerate(address_dict):
        labels = [label[1] for label in address_dict[address]]
        missing_labels = address_labels.difference(set(labels))
        missing_tuples = [(' ', label) for label in missing_labels]
        all_labels = address_dict[address] + missing_tuples
        reorder_labels = [(label[1], label[0]) for label in all_labels]
        address_dict[list(address_dict.keys())[i]] = reorder_labels

    return address_dict

# break up street address into address1 & address2
def split_streetAddress(labels_dict):
    address1_lst = ['AddressNumber', 'StreetNamePreDirectional', 'StreetName', 'StreetNamePostType', 'StreetNamePostDirectional'] 
    address2_lst = ['OccupancyType', 'OccupancyIdentifier']

    address1_labels = {k: v for k, v in labels_dict.items() if k in address1_lst}
    address1 = ''
    for k, v in address1_labels.items():
        address1 = address1 + v + ' '

    address2_labels = {k: v for k, v in labels_dict.items() if k in address2_lst}
    address2 = ''
    for k, v in address2_labels.items():
        address2 = address2 + v + ' '

    return (address1, address2)

# break up po box into address1 & address2
def split_poBox(labels_dict):
    address1_lst = ['SubaddressType', 'SubaddressIdentifier']
    address2_lst = ['USPSBoxType',  'USPSBoxID']

    address1_labels = {k: v for k, v in labels_dict.items() if k in address1_lst}
    address1 = ''
    for k, v in address1_labels.items():
        address1 = address1 + v + ' '

    address2_labels = {k: v for k, v in labels_dict.items() if k in address2_lst}
    address2 = ''
    for k, v in address2_labels.items():
        address2 = address2 + v + ' '
        
    return (address1, address2)

# create fulladdress, address1 & address2 depending on street type
def determine_streetType(address_dict, address, label_dict, labels):
    street_dict = { k: label_dict[k] for k in labels }
    full_address = ''
    full_street_labels = {k: v for k, v in street_dict.items() if v != ' '}
    for k, v in full_street_labels.items():
        full_address = full_address + v + ' '
    full_address_pair = [('FullAddress', full_address.rstrip())]

    if 'OccupancyType' in list(full_street_labels.keys()):
        addresses = split_streetAddress(full_street_labels)
        address1_pair = [('Address1', addresses[0])]
        address2_pair = [('Address2', addresses[1])]
    
    elif 'USPSBoxType' in list(full_street_labels.keys()):
        addresses = split_poBox(full_street_labels)
        address1_pair = [('Address1', addresses[0])]
        address2_pair = [('Address2', addresses[1])]

    else:
        address1_pair = [('Address1', full_address)]
        address2_pair = [('Address2', '')]

    address_dict[address] += full_address_pair
    address_dict[address] += address1_pair
    address_dict[address] += address2_pair
    address_dict[address] = dict(address_dict[address])
    
    return address_dict[address]

def create_addressCols(n):

    street_labels = ['AddressNumber', 'StreetNamePreDirectional', 'StreetName', 'StreetNamePostType', 'StreetNamePostDirectional', 'OccupancyType', 'OccupancyIdentifier']
    po_labels = ['SubaddressType', 'SubaddressIdentifier', 'USPSBoxType',  'USPSBoxID']
    landmark_labels = ['LandmarkName']
    recipient_labels = ['Recipient']

    address_dict = fillEmpty_keys(n)
    for address in address_dict:
        label_dict = dict(address_dict[address])
        
        if label_dict['AddressNumber'] != ' ':
            address_dict[address] = determine_streetType(address_dict, address, label_dict, street_labels)

        elif label_dict['SubaddressType'] != ' ':
            address_dict[address] = determine_streetType(address_dict, address, label_dict, po_labels)

        elif label_dict['LandmarkName'] != ' ':
            address_dict[address] = determine_streetType(address_dict, address, label_dict, landmark_labels)

        elif label_dict['Recipient'] != ' ':
            address_dict[address] = determine_streetType(address_dict, address, label_dict, recipient_labels)
        
    return address_dict

# create dataframe of street address labels
def addressDF(n):
    addresses = create_addressCols(n)

    to_keep = [k for k, v in list(addresses.items()) if isinstance(v, dict)]
    filter_address_dict = { k: addresses[k] for k in to_keep }

    df = pd.DataFrame.from_dict(filter_address_dict)
    df = df.T.reset_index().rename(columns={'PlaceName':'City', 'StateName':'State'})
    df = df[['FullAddress', 'Address1', 'Address2', 'City', 'State', 'ZipCode']]
    return df

# create random dates for mail + transaction
def generate_random_dates(year, k, seed_num):
    # initialize dates ranges 
    date1, date2 = date(year, 1, 1), date(year, 12, 31)
    
    # calculate number of days between dates
    dates_between = date2 - date1
    total_days = dates_between.days
    
    random.seed(seed_num)
    date_lst = [date1 + timedelta(days=random.randrange(total_days)) for i in range(k)]
    
    return date_lst

# create transaction dataframe
def transaction_files(n, seed_num):
    random.seed(seed_num)

    df = addressDF(n)

    len_df = len(df)

    revenues = [round(random.uniform(1.00, 1000.00), 2) for i in range(len_df)]

    dates = generate_random_dates(2021, len_df, seed_num)

    orderNumbers = [random.randint(10000000,99999999) for i in range(len_df)]

    userIds = ['U' + str(random.randint(1000000,9999999)) for i in range(len_df)]

    firstTime = [random.randint(0, 1) for i in range(len_df)]

    df['revenue'] = revenues

    df['date'] = dates

    df['Order Number'] = orderNumbers

    df['User ID'] = userIds

    df['First Time Order'] = firstTime

    return df

def mail_files(n, seed_num):
    random.seed(seed_num)

    df = addressDF(n)

    len_df = len(df)

    mail_split = int(len_df/2)

    mom_lst = ['MOM' + str(random.randint(1000000,9999999)) for i in range(mail_split)]

    dog_lst = ['DOG' + str(random.randint(1000000,9999999)) for i in range(mail_split)]

    mail_lst = mom_lst + dog_lst

    df['mailkey'] = mail_lst

    return df

def create_holdouts(n, seed_num):
    df = mail_files(n,seed_num)
    
    n = n/4

    holdout_df = df.sample(int(n), random_state=int(seed_num))
    holdout_lst = ['H-' + mailkey for mailkey in holdout_df['mailkey']]
    holdout_df['mailkey'] = holdout_lst

    return holdout_df

