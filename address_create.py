import pandas as pd
from faker import Faker
import usaddress
from collections import defaultdict
import random
from datetime import date, timedelta

# Creates a list of fake addresses using Faker.
# n = number of addresses to create
# faker_seed = seed number of address creation
def create_addresses(n,faker_seed):
    fake = Faker()
    Faker.seed(faker_seed)
    address_list = [fake.address() for _ in range(n)]
    return address_list


# Parses addresses into labels and stores each pair as a dictionary. Full address is the key. Parsed labels are the values.
# n = number of addresses to create
# faker_seed = seed number of address creation
def parse_address(n, faker_seed):
    address_dict = defaultdict(list)

    fake_addresses = create_addresses(n, faker_seed)

    # create dict of addresses and parsed sections
    for address in fake_addresses:
        parsed_address = usaddress.parse(address)
        for parsed in parsed_address:
            address_dict[address].append(parsed)

    return address_dict

# Cleans parsed address labels by removing new lines and commas.
# n = number of addresses to create
# faker_seed = seed number of address creation
def clean_labels(n, faker_seed):
    address_dict = parse_address(n, faker_seed)
    for address in address_dict:
        for i, label in enumerate(address_dict[address]):
            label_lst = list(label)
            label_lst[0] = label_lst[0].rstrip().rstrip(',')
            label_tuple = tuple(label_lst)
            address_dict[address][i] = label_tuple

    return address_dict

# Combines values if they share the same label.
# n = number of addresses to create
# faker_seed = seed number of address creation
def combineConsecutive_labels(n, faker_seed):
    address_dict = clean_labels(n, faker_seed)
    for address in address_dict:
        address_labels = address_dict[address]
        for i, label in enumerate(address_labels):
            if address_labels[i][1] == address_labels[i-1][1]:
                new_lst =  address_labels[:i-1] + address_labels[i+1:]
                new_lst.append((address_labels[i-1][0] + ' ' + address_labels[i][0], label[1]))
                address_dict[address] = new_lst
    return address_dict

# Adds missing labels to the dictionary & re-orders tuple elements so the label comes first & value comes second
# n = number of addresses to create
# faker_seed = seed number of address creation
def fillEmpty_keys(n, faker_seed):
    address_labels = {'AddressNumber', 'StreetName', 'StreetNamePostType', 'OccupancyType',
                    'OccupancyIdentifier', 'PlaceName', 'StateName', 'ZipCode', 'StreetNamePostDirectional', 
                    'StreetNamePreDirectional', 'SubaddressType', 'SubaddressIdentifier', 'USPSBoxType', 
                    'USPSBoxID', 'Recipient', 'LandmarkName'}

    address_dict = combineConsecutive_labels(n, faker_seed)

    for i, address in enumerate(address_dict):
        labels = [label[1] for label in address_dict[address]]
        missing_labels = address_labels.difference(set(labels))
        missing_tuples = [(' ', label) for label in missing_labels]
        all_labels = address_dict[address] + missing_tuples
        reorder_labels = [(label[1], label[0]) for label in all_labels]
        address_dict[list(address_dict.keys())[i]] = reorder_labels

    return address_dict

# Breaks up street address into address1 & address2 labels 
# labels_dict = complete street address labels
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

# Breaks up po box into address1 & address2 labels 
# labels_dict = complete po box labels
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

# Creates full address, address1 & address2 values depending on street type
# address_dict = dictionary of addresses as the key and their parsed labels as the values
# address = key from address_dict
# label_dict = dictionary of parsed address labels
# labels = list of address labels, dependant on street type
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

# Updates address_dict to have correct parsed street labels depending on street type
# n = number of addresses to create
# faker_seed = seed number of address creation
def create_addressCols(n, faker_seed):
    street_labels = ['AddressNumber', 'StreetNamePreDirectional', 'StreetName', 'StreetNamePostType', 'StreetNamePostDirectional', 'OccupancyType', 'OccupancyIdentifier']
    po_labels = ['SubaddressType', 'SubaddressIdentifier', 'USPSBoxType',  'USPSBoxID']
    landmark_labels = ['LandmarkName']
    recipient_labels = ['Recipient']

    address_dict = fillEmpty_keys(n, faker_seed)
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

# Creates dataframe of fake addresses and all of its labels 
# n = number of addresses to create
# faker_seed = seed number of address creation
def addressDF(n, faker_seed):
    addresses = create_addressCols(n, faker_seed)

    to_keep = [k for k, v in list(addresses.items()) if isinstance(v, dict)]
    filter_address_dict = { k: addresses[k] for k in to_keep }

    df = pd.DataFrame.from_dict(filter_address_dict)
    df = df.T.reset_index().rename(columns={'PlaceName':'City', 'StateName':'State'})
    df = df[['FullAddress', 'Address1', 'Address2', 'City', 'State', 'ZipCode']]
    return df

# Create random dates for mail + transaction files
# year = current year
# k = number of dates to create
# seed_num = seed number for randomly sampling dates
def generate_random_dates(year, k, seed_num):
    # initialize dates ranges 
    date1, date2 = date(year, 1, 1), date(year, 12, 31)
    
    # calculate number of days between dates
    dates_between = date2 - date1
    total_days = dates_between.days
    
    random.seed(seed_num)
    date_lst = [date1 + timedelta(days=random.randrange(total_days)) for i in range(k)]
    
    return date_lst

# Creates transactions dataframe. Columns include address labels, revenue, date, order number, user id, first time order flag
# n = number of addresses to create
# seed_num = seed number for randomly sampling dates
# faker_seed = seed number of address creation
def transaction_files(n, seed_num, faker_seed):
    random.seed(seed_num)

    df = addressDF(n, faker_seed)

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

# Creates mailing dataframe by randomly sampling from transactions. Mailkey column is also created. 
# n = number of addresses to create
# seed_num = seed number for randomly sampling dates
# faker_seed = seed number of address creation
def mail_files(n, seed_num, faker_seed):
    random.seed(seed_num)

    df = addressDF(n, faker_seed)

    if len(df) % 2 != 0:
        df.drop(df.tail(1).index,inplace=True)

    len_df = len(df)

    mail_split = int(len_df/2)

    mom_lst = ['MOM' + str(random.randint(1000000,9999999)) for i in range(mail_split)]

    dog_lst = ['DOG' + str(random.randint(1000000,9999999)) for i in range(mail_split)]

    mail_lst = mom_lst + dog_lst

    df['mailkey'] = mail_lst

    return df

# Creates holdouts dataframe by randomly sampling from mailing dataframe and creating holdout mail keys
# n = number of addresses to create
# seed_num = seed number for randomly sampling dates
# faker_seed = seed number of address creation
def create_holdouts(n, seed_num, faker_seed):
    df = mail_files(n,seed_num, faker_seed)
    
    n = n/4

    holdout_df = df.sample(int(n), random_state=int(seed_num))
    holdout_lst = ['H-' + mailkey for mailkey in holdout_df['mailkey']]
    holdout_df['mailkey'] = holdout_lst

    return holdout_df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='number of addresses to create (int)')
    parser.add_argument('seed_num', type=int, help='seed number to randomly sample addresses (int')
    parser.add_argument('faker_seed', type=int, help='seed number for address creation (int)')
    parser.add_argument('transaction_name', help='name of transaction csv file (string)')
    parser.add_argument('mailing_name', help='name of mailing csv file (string)')
    parser.add_argument('holdout_name', help='name of holdout csv file (string)')
    args = parser.parse_args()

    transaction_df = transaction_files(args.n, args.seed_num, args.faker_seed)
    mailing_df = mail_files(args.n, args.seed_num, args.faker_seed)
    holdout_df = create_holdouts(args.n, args.seed_num, args.faker_seed)

    transaction_df.to_csv(args.transaction_name, index=False)
    mailing_df.to_csv(args.mailing_name, index=False)
    holdout_df.to_csv(args.holdout_name, index=False)