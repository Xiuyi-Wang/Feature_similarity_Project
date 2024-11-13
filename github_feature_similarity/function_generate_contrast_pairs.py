def generate_contrast_pairs(networks):
    """
    generate contrast pairs for FC between networks_names
    :param networks: a list, including the networks_names names
    :return: contrast pairs, a nested list

    examples:
    networks_names = ['A','B','C','D','E','F']
    contrast_pairs = generate_contrast_pairs(networks_names)
    """

    # todo you ignored CBCA
    contrast_pairs = []

    for i in range(len(networks)-1):
        Bs = None
        A = networks[i]

        if i == 0:
            B = networks[i+1]

        elif i > 0 and i < 2:
            B = networks[i-1]

        elif i > 1:
            Bs = []

            for k in range(1,i+1):
                B = networks[i - k]
                Bs.append(B)

        for j in range(i+1,len(networks)):
            C = networks[j]

            if Bs == None:

                contrast_pair = [A,B,A,C]
                contrast_pairs.append(contrast_pair)
            else:
                for B in Bs:
                    contrast_pair = [A, B, A, C]
                    contrast_pairs.append(contrast_pair)

    # remove the first element
    contrast_pairs.pop(0)

    return contrast_pairs

def generate_contrast_pairs_all (networks):
    """
    :param networks: is a list
    :return:
    """
    from itertools import combinations

    # generate combinations 1
    network_pairs = list(combinations(networks, 2))
    # generate combinations of combinations 1
    network_pairs_pairs = list(combinations(network_pairs, 2))

    # remove the combinations that do not have one common element
    networks_pairs_final = []

    for network_pair in network_pairs_pairs:

        # make sure has one common element in the first part and second part
        if network_pair[0][0] in network_pair[1] or network_pair[0][1] in network_pair[1]:

            if network_pair[0][1] in network_pair[1]:
                # adjust the order make sure the common element is the first
                pair = [network_pair[0][1] , network_pair[0][0] , network_pair[1][0] , network_pair[1][1]]

            else:
                pair = list(network_pair[0] + network_pair[1])

            # make sure the third element is equal to first element
            if pair[0] == pair[3]:
                new_pair = [pair[0] , pair[1] , pair[3] , pair[2]]

            else:
                new_pair = pair

            networks_pairs_final.append(new_pair)

    return networks_pairs_final



# # to test the function
#
# networks_names = ['A','B','C','D','E','F']
# contrast_pairs = generate_contrast_pairs(networks_names)
# print (contrast_pairs)
#
# networks_names = ['VisualA', 'VisualB',   'SomMotA', 'SomMotB',
#             'SalVenAttnA', 'SalVenAttnB']
# networks_pairs_final = generate_contrast_pairs_all (networks_names)
#
# print (networks_pairs_final)