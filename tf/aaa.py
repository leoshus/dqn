def hello_pci(f):
    print("Hello PCI")
    return f


@hello_pci
def message_1():
    print("Message_1")


@hello_pci
def message_2():
    print("Message_2")


if __name__ == "__main__":
    message_1()
    message_2()
    '北京'.encode()
    '北京'.encode().decode()
    '北京'.decode()
