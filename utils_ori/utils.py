import torch 
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import torch.nn.functional as F

############ Evaluation ############
def eval(model, dataset):
    acc_list =[]
    tn_total = 0
    fp_total = 0
    fn_total = 0
    tp_total = 0
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            output = model(batch['keypoints'])
            y_pred = np.rint(output.reshape(-1).detach().numpy())
            acc = accuracy_score(batch['output'], y_pred)
            cm = confusion_matrix(batch['output'], y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            acc_list.append(acc)
            tn_total += tn
            fp_total += fp
            fn_total += fn
            tp_total += tp

    return {"accuracy": float(np.array(acc_list).mean()),"tn": int(tn_total),"fp": int(fp_total) , "fn": int(fn_total), "tp": int(tp_total)}

def eval_seq(model, dataset, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    token_acc_list =[]
    join_acc_list = []
    preds_list, labels_list = [], []
    output_ids, output_preds, output_labels = [], [], []
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            batch_size = batch['keypoints'].size(0)
            keypoints, batch_labels = batch['keypoints'].to(device), batch['output'].to(device)
            mask = batch['mask'].to(device)
            output = model(keypoints, mask)

            output_ids.extend(batch['ids'])          
            for labels, m in zip(batch_labels, mask):
                labels_list += (labels[m==True].tolist())
                output_labels.append(labels[m==True].tolist())
            
            batch_preds = torch.argmax(output,dim=1)
            ## reshape: (batch * seq_len) ---> (batch, seq_len)
            batch_preds = batch_preds.view(batch_size, -1)

            for preds, m in zip(batch_preds, mask):
                preds_list += (preds[m==True].tolist())
                output_preds.append(preds[m==True].tolist())

    preds_list_tensor = torch.tensor(preds_list, dtype=float)
    labels_list_tensor = torch.tensor(labels_list, dtype=float)
    token_acc = (preds_list_tensor == labels_list_tensor).sum()/len(preds_list_tensor)
    mse = F.mse_loss(labels_list_tensor, preds_list_tensor)
    if mode == "test":
        print("************")
        print(batch_labels[0].tolist())
        print(batch_preds[0].tolist())
        print("************")
        print("TOKEN ACCURACY: {:.1%}".format(token_acc))
    print("MSE: {:.2f}".format(mse))
    print("================= END OF EVALUATION ================")
    return {"accuracy": token_acc.detach().item(),"mse": mse.detach().item(), "ids": output_ids, "predictions": output_preds, "labels": output_labels}

def eval_crf(model, dataset, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds_list, labels_list = [], []
    output_ids, output_preds, output_labels = [], [], []
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            keypoints, batch_labels = batch['keypoints'].to(device), batch['output'].to(device)
            mask = batch['mask'].to(device)
            #### output: list of predictions
            batch_preds = model(keypoints, mask)
            
            output_ids.extend(batch['ids'])
            output_preds.extend(batch_preds)
            for preds in batch_preds:
                preds_list += preds
            for labels, m in zip(batch_labels, mask):
                labels_list += labels[m==True].tolist()
                output_labels.append(labels[m==True].tolist())
    
    preds_list_tensor = torch.tensor(preds_list, dtype=float)
    labels_list_tensor = torch.tensor(labels_list, dtype=float)
    token_acc = (preds_list_tensor == labels_list_tensor).sum()/len(preds_list_tensor)
    mse = F.mse_loss(labels_list_tensor, preds_list_tensor)
    if mode == "test":
        print("************")
        print(batch_labels[0].tolist())
        print(batch_preds[0])
        print("************")
        print("TOKEN ACCURACY: {:.1%}".format(token_acc))
    print("MSE: {:.2f}".format(mse))
    print("================= END OF EVALUATION ================")
    return {"accuracy": token_acc.detach().item(),"mse": mse.detach().item(),"ids": output_ids, "predictions": output_preds, "labels": output_labels}

def nll_loss(predict, y):
    PAD_IDX = 4
    # convert y from (batch_size, max_len) to (batch_size * max_len)
    y = y.contiguous().view(-1)
    if_padded = (y < PAD_IDX).float()
    total_token = int(torch.sum(if_padded).item())
    # predict: (batch_size * max_len, num_class)
    predict = predict[range(predict.size(0)), y]* if_padded
    ce = -torch.sum(predict) / total_token
    return ce

# def eval_seq2seq(model, dataset, mode):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     token_acc_list =[]
#     preds_list, labels_list = [], []
#     output_ids, output_preds, output_labels = [], [], []
#     loss_list = []
#     for it, batch in enumerate(dataset):
#         with torch.no_grad():
#             batch_size = batch['keypoints'].size(0)
            
#             #---
#             keypoints, labels, labels_embeddings = batch['keypoints'].to(device), batch['output'].to(device), batch['tgt_embeddings'].to(device)
            
#             labels_input = labels[:, :-1]
#             # labels_embeddings_input = labels_embeddings[:, :-1, :]
#             labels_expected = labels[:, 1:]
            
#             # create masks
#             src_padding_mask = batch['mask'].to(device)
#             tgt_mask, _ = model.create_mask(labels_input)
#             tgt_mask = tgt_mask.to(device)
#             tgt_padding_mask = src_padding_mask[:, :-1]
             
#             # generate output
#             output = model(keypoints, labels_input, tgt_mask, src_padding_mask, tgt_padding_mask)

#             #calculate loss
#             loss = nll_loss(output, labels_expected)
#             loss_list.append(loss.detach().item())
#             #---

#             output_ids.extend(batch['ids'])
#             _, labels_expected_padding_mask = model.create_mask(labels_expected)         
#             for labels, m in zip(labels_expected, labels_expected_padding_mask):
#                 labels_list += (labels[m==False].tolist())
#                 output_labels.append(labels[m==False].tolist())
            
#             batch_preds = torch.argmax(output,dim=1)
#             ## reshape: (batch * seq_len) ---> (batch, seq_len)
#             batch_preds = batch_preds.view(batch_size, -1)

#             for preds, m in zip(batch_preds, labels_expected_padding_mask):
#                 preds_list += (preds[m==False].tolist())
#                 output_preds.append(preds[m==False].tolist())

#     preds_list_tensor = torch.tensor(preds_list, dtype=float)
#     labels_list_tensor = torch.tensor(labels_list, dtype=float)
#     token_acc = (preds_list_tensor == labels_list_tensor).sum()/len(preds_list_tensor)
#     print("************")
#     print(output_labels[0])
#     print(output_preds[0])
#     print("************")
#     print("TOKEN ACCURACY: {:.1%}".format(token_acc))
#     print("Loss: {:.4f}".format(sum(loss_list)/len(loss_list)))
#     print("================= END OF EVALUATION ================")
#     return {"accuracy": token_acc.detach().item(), "ids": output_ids, "predictions": output_preds, "labels": output_labels}

def eval_seq2seq(model, dataset, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    token_acc_list =[]
    preds_list, labels_list = [], []
    output_ids, output_preds, output_labels = [], [], []
    loss_list = []
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            batch_size = batch['keypoints'].size(0)
            
            #---
            keypoints, labels = batch['keypoints'].to(device), batch['output'].to(device)
            
            labels_input = labels[:, :-1]
            # labels_embeddings_input = labels_embeddings[:, :-1, :]
            labels_expected = labels[:, 1:]
            
            # create masks
            tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(keypoints, labels)
            tgt_mask, src_padding_mask, tgt_padding_mask = tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
             
            # generate output
            output = model(keypoints, labels_input, tgt_mask, src_padding_mask, tgt_padding_mask)

            #calculate loss
            loss = nll_loss(output, labels_expected)
            loss_list.append(loss.detach().item())
            #---

            output_ids.extend(batch['ids'])
            PAD_IDX = 4
            labels_expected_padding_mask = (labels_expected == PAD_IDX)         
            for labels, m in zip(labels_expected, labels_expected_padding_mask):
                labels_list += (labels[m==False].tolist())
                output_labels.append(labels[m==False].tolist())
            
            batch_preds = torch.argmax(output,dim=1)
            ## reshape: (batch * seq_len) ---> (batch, seq_len)
            batch_preds = batch_preds.view(batch_size, -1)

            for preds, m in zip(batch_preds, labels_expected_padding_mask):
                preds_list += (preds[m==False].tolist())
                output_preds.append(preds[m==False].tolist())

    preds_list_tensor = torch.tensor(preds_list, dtype=float)
    labels_list_tensor = torch.tensor(labels_list, dtype=float)
    token_acc = (preds_list_tensor == labels_list_tensor).sum()/len(preds_list_tensor)
    print("************")
    print(output_labels[0])
    print(output_preds[0])
    print("************")
    print("TOKEN ACCURACY: {:.1%}".format(token_acc))
    print("Loss: {:.4f}".format(sum(loss_list)/len(loss_list)))
    print("================= END OF EVALUATION ================")
    return {"accuracy": token_acc.detach().item(), "ids": output_ids, "predictions": output_preds, "labels": output_labels, "loss": sum(loss_list)/len(loss_list)}