import synapseclient 
 
syn = synapseclient.Synapse() 
syn.login('ianva','16387302') 
 
# Obtain a pointer and download the data 
syn3379050 = syn.get(entity='syn3379050') 
 