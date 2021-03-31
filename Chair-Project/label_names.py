with open('mscoco_labels.names.txt', "r") as f:
    i = 0
    for line in file:
        if line == "":
            continue
        if line == "person" or line == "chair":
            print(i+1)
        i += 1

        
            
    
