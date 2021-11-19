from faker import Faker

# create n number of fake addresses
def create_fakeAddress(n):
    fake = Faker()
    Faker.seed(4321)
    address_list = [fake.address() for _ in range(n)]
    return address_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n', help='number of addresses to create')
    #parser.add_argument('output_file', help='cleaned data file (CSV)')
    args = parser.parse_args()

    address_list = create_fakeAddress(args.n)