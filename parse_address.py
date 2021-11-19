# take txtfile of addresses and parse address labels into a dictionary
import usaddress

def parse_address(n, txtfile):
    address_dict = defaultdict(list)

    fake_addresses = [address.strip() for address in txtfile]
    #fake_addresses = create_addresses(n)

    for address in fake_addresses:
        parsed_address = usaddress.parse(address)
        for parsed in parsed_address:
            address_dict[address].append(parsed)

    return address_dict

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n', help='number of addresses to create')
    parser.add_argument('input_file', help='txt file of addresses')
    #parser.add_argument('output_file', help='combined data file (CSV)')
    args = parser.parse_args()

    address_dict = parse_address(args.n, args.input_file)


