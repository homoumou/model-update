# model-update
SWAT
SWaT is a key asset for researchers aiming at the design of safe and secure cyber physical systems (CPS.) The testbed consists of a modern six-stage water treatment process that closely mimics a real world treatment plant. Stage 1 of the physical process begins by taking in raw water, followed by chemical dosing (Stage 2), filtering it through an Ultrafiltration (UF) system (Stage 3), dechlorination using UV lamps (Stage 4), and then feeding it to a Reverse Osmosis (RO) system (Stage 5). A backwash process (Stage 6) cleans the membranes in UF using the RO permeate. SWaT dataset consists of 11 days of continuous operation – of which 7 days’ worth of data was collected under normal operation while 4 days’ worth of data was collected with attack scenarios.
![image](https://user-images.githubusercontent.com/41387221/159193415-63acb852-a3ac-4fa5-acb8-31385597d7e7.png)
![image](https://user-images.githubusercontent.com/41387221/159193426-f4c52931-32c6-4c74-a69f-d40c4b278971.png)

 



The dataset contains 79 features with a total of 14,996 data
 ![image](https://user-images.githubusercontent.com/41387221/159193427-3f733ee0-77e6-480e-af71-94acb7ef2a41.png)


Categorical variable:
 ![image](https://user-images.githubusercontent.com/41387221/159193434-6d3d29da-9455-4819-a993-b8415b079c5a.png)


Label value counts:
 ![image](https://user-images.githubusercontent.com/41387221/159193438-b53976b2-7561-491b-95fc-e744d6254bff.png)


Data preprocessing:
We encoder Categorical variable by using labelEncoder
 ![image](https://user-images.githubusercontent.com/41387221/159193441-85bccf58-5a6c-4ead-b4f6-796ff66bd515.png)


Normalized the dataset by using MinMaxScaler
 ![image](https://user-images.githubusercontent.com/41387221/159193448-96a47930-ff6c-489b-aa02-f6f100deb994.png)

Split dataset as training dataset and validation dataset
 ![image](https://user-images.githubusercontent.com/41387221/159193453-a40418d6-b639-49b9-9b39-2ff5f283421d.png)


Training Process
Considering the timing serialization of SWAT, we choose LSTM as the training model.
For our model, we use 1 LSTM layer, 1 dropout layer and 1 fully connect layer, for activation function we will use the sigmoid function.
![image](https://user-images.githubusercontent.com/41387221/159193458-ea7f7ff3-5c36-476e-881c-8eb42edba873.png)

 


Dataloader:
We are packing data as TensorDataset and divide as different batch, the batch size is 60.
 ![image](https://user-images.githubusercontent.com/41387221/159193461-3bc48c98-687f-4b26-850c-6534e1dd2064.png)





For the training
Epochs number: 20
 ![image](https://user-images.githubusercontent.com/41387221/159193466-2772c3b1-f09f-41bb-a862-0a9573285129.png)

















Evaluation :
After 20 epoch training, the final accuracy can reach 0.95 on valid dataset and 0.933 on training dataset

 ![image](https://user-images.githubusercontent.com/41387221/159193477-23380cd7-34c5-45db-bbb6-ea3182cda7de.png)
![image](https://user-images.githubusercontent.com/41387221/159193485-d80c729f-48d3-4853-8880-26b9c0cc036d.png)


 



streamlines continual model updating: 
To address concept drift, we introduce the continual model update framework.
The model update strategies are inspirited by online learning concept. The framework is composed by two components. 
![image](https://user-images.githubusercontent.com/41387221/159193494-aac857df-0e97-4d3e-a905-e0a8d24bb0ca.png)

 
Runtime Profiler:
This component is used for logs and profiles the training time for each model. According to the dataset size, we can predict their approximate training time by using learning regression.
![image](https://user-images.githubusercontent.com/41387221/159193510-e734e742-c364-4345-9738-cdabd124435d.png)


Update Controller:
Update Controller is used to decides when to perform model update.
To finding most suitable timing for model update, we calculate the Data Incorporation Latency. Data Incorporation Latency indicates how fast the data is incorporated into the model, and we measure the delay between the arrival of each sample and the time at which that sample is incorporated.
Suppose we have m data samples arrive in sequence at time a_1,…., a_m. The system performs n model updates, where the i-th update starts at s_i and takes t_i time to complete. We have D_i be the set of data samples incorporated by the i-th update, which arrived after the (i-1)-th update starts and before the i-th:
![image](https://user-images.githubusercontent.com/41387221/159193519-4a7a55e9-9759-4ead-acc9-bb6f1820ec7b.png)

Since all sample in D_i get incorporated after the i-th update completes, the cumulative latency is computed as 
![image](https://user-images.githubusercontent.com/41387221/159193608-46bd878f-31a6-4a8e-8941-f00ce85125e3.png)

Summing up L_i over all n updates, we obtain the data incorporation latency
![image](https://user-images.githubusercontent.com/41387221/159193577-05ca05d2-56e7-4219-99c8-848fa73b178a.png)

To minimum the training cost, we propose a cost-aware policy for fast data incorporation at low training cost. we introduce “knob” parameter w, meaning, for every unit of training cost it spends, it expects the data incorporation latency to be reduced by w. In this model, latency Li and cost τi are “exchangeable” and are hence unifed as one objective, which we call latency-cost sum, i.e.
![image](https://user-images.githubusercontent.com/41387221/159193595-09371259-1243-434c-846a-d84794825e56.png)

To compute the maximum latency reduction, we develop cost-aware model update strategy. 
 ![image](https://user-images.githubusercontent.com/41387221/159193538-515135db-73f9-4efc-aa8f-62c74471104c.png)

