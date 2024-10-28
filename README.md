
# Poster: Lambda: Learning Matchable Prior For Entity Alignment with Unlabeled Dangling Cases

![Uploading Poster_NeurIPS_01.png…]()

# Program Directory Structure

 ('extract_' files represent files that are generated after the program is run):

```
├─PU-DED
│  Dual_dangling.py
│  Dual_dangling_auto.py
│  evaluate.py
│  layer.py
│  test.py
│  utils.py
│  └──data
│     ├─DBP15K
│     │  ├─fr_en
│     │  │        ent_ids_1
│     │  │        ent_ids_2
│     │  │        ref_ent_ids 
│     │  │        triples_1
│     │  │        triples_2 
│     │  │
│     │  ├─ja_en
│     │  │        ent_ids_1
│     │  │        ent_ids_2
│     │  │        ref_ent_ids 
│     │  │        triples_1
│     │  │        triples_2 
│     │  │
│     │  └─ zh_en
│     │           ent_ids_1
│     │           ent_ids_2
│     │           ref_ent_ids 
│     │           triples_1
│     │           triples_2
│     │
│     ├─DBP2.0-minus
│     │  ├─DBP2.0-JA-EN-minus_en
│     │  │        ent_ids_1
│     │  │        ent_ids_2
│     │  │        ref_ent_ids 
│     │  │        triples_1
│     │  │        triples_2 
│     │  │
│     │  └─ DBP2.0-ZH-EN-minus_en
│     │           ent_ids_1
│     │           ent_ids_2
│     │           ref_ent_ids 
│     │           triples_1
│     │           triples_2
│     │
│     ├─DBP2.0-plus
│     │  ├─DBP2.0-JA-EN-plus_en
│     │  │        ent_ids_1
│     │  │        ent_ids_2
│     │  │        ref_ent_ids 
│     │  │        triples_1
│     │  │        triples_2 
│     │  │
│     │  └─ DBP2.0-ZH-EN-plus_en
│     │           ent_ids_1
│     │           ent_ids_2
│     │           ref_ent_ids 
│     │           triples_1
│     │           triples_2
│     │
│     ├─GA-DBP15K
│     │  ├─GA-EN
│     │  │  ├─0.1_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  ├─0.15_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  ├─0.2_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  └─0.25_en
│     │  │           ent_ids_1
│     │  │           ent_ids_2
│     │  │           ref_ent_ids 
│     │  │           triples_1
│     │  │           triples_2
│     │  │
│     │  ├─GA-FR
│     │  │  ├─0.1_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  ├─0.15_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  ├─0.2_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  └─0.25_en
│     │  │           ent_ids_1
│     │  │           ent_ids_2
│     │  │           ref_ent_ids 
│     │  │           triples_1
│     │  │           triples_2
│     │  │
│     │  ├─GA-JA
│     │  │  ├─0.1_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  ├─0.15_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  ├─0.2_en
│     │  │  │        ent_ids_1
│     │  │  │        ent_ids_2
│     │  │  │        ref_ent_ids 
│     │  │  │        triples_1
│     │  │  │        triples_2
│     │  │  │
│     │  │  └─0.25_en
│     │  │           ent_ids_1
│     │  │           ent_ids_2
│     │  │           ref_ent_ids 
│     │  │           triples_1
│     │  │           triples_2
│     │  │
│     │  └─ GA-ZH
│     │     ├─0.1_en
│     │     │        ent_ids_1
│     │     │        ent_ids_2
│     │     │        ref_ent_ids 
│     │     │        triples_1
│     │     │        triples_2
│     │     │
│     │     ├─0.15_en
│     │     │        ent_ids_1
│     │     │        ent_ids_2
│     │     │        ref_ent_ids 
│     │     │        triples_1
│     │     │        triples_2
│     │     │
│     │     ├─0.2_en
│     │     │        ent_ids_1
│     │     │        ent_ids_2
│     │     │        ref_ent_ids 
│     │     │        triples_1
│     │     │        triples_2
│     │     │
│     │     └─0.25_en
│     │              ent_ids_1
│     │              ent_ids_2
│     │              ref_ent_ids 
│     │              triples_1
│     │              triples_2
│     │
│     ├─GA16K
│     │        ref_ent_ids
│     │        ref_pairs
│     │        sup_ent_ids
│     │        sup_pairs
│     │        triples_1
│     │        triples_2 
│     │
│     └─dbp2.0
│        ├─fr_en
│        │        ent_ids_1
│        │        ent_ids_2
│        │       ref_ent_ids 
│        │        triples_1
│        │        triples_2 
│        │        extract_node_1
│        │        extract_node_2
│        │        extract_node
│        │
│        ├─ja_en
│        │        ent_ids_1
│        │       ent_ids_2
│        │        ref_ent_ids 
│        │        triples_1
│        │        triples_2 
│        │        extract_node_1
│        │        extract_node_2
│        │        extract_node
│        │
│        └─ zh_en
│                 ent_ids_1
│                 ent_ids_2
│                 ref_ent_ids 
│                 triples_1
│                 triples_2 
│                 extract_node_1
│                 extract_node_2
│                 extract_node
├─EA
│  Dual_align.py
│  Dual_joint.py
│  layer.py
│  test.py
│  utils.py
│  └──data
│     ├─dbp2.0
│     │  ├─fr_en
│     │  │        ent_ids_1
│     │  │        ent_ids_2
│     │  │        rel_ids_1
│     │  │        rel_ids_2
│     │  │        ref_ent_ids 
│     │  │        triples_1
│     │  │        triples_2 
│     │  │        extract_node_1
│     │  │        extract_node_2
│     │  │        extract_node
│     │  │
│     │  ├─ja_en
│     │  │        ent_ids_1
│     │  │        ent_ids_2
│     │  │        rel_ids_1
│     │  │        rel_ids_2
│     │  │        ref_ent_ids 
│     │  │        triples_1
│     │  │        triples_2 
│     │  │        extract_node_1
│     │  │        extract_node_2
│     │  │        extract_node
│     │  │
│     │  └─ zh_en
│     │           ent_ids_1
│     │           ent_ids_2
│     │           rel_ids_1
│     │           rel_ids_2
│     │           ref_ent_ids 
│     │           triples_1
│     │           triples_2 
│     │           extract_node_1
│     │           extract_node_2
│     │           extract_node
│     │
│     └─GA16K
│             ref_ent_ids
│             sup_ent_ids
│             ref_pairs
│             sup_pairs
│             triples_1
│             triples_2
└──readme.md
```


# File Instructions：

#### PU-DED: This folder contains the dangling detection component;

##### 	│  Dual_dangling.py	

​	Used to classify dangling and matchable entities. It includes the whole process of data input, training and testing of the model. It requires the use of a function interface written in the following files.

##### 	│  Dual_dangling_auto.py	

​ The iPULE algorithm version of Dual_dangling.py.

##### 	│  evaluate.py

​	A collection of functional interfaces written to test the accuracy of the model.

##### 	│  layer.py

​	A collection of functional interfaces written to constitute the component of the model.

##### 	│  utils.py

​	A collection of functional interfaces written for data input.

#### EA: This folder contains the Entity Alignment component;

##### 	│  Dual_align.py

​	Used to employ entity alignment task on different datasets. It includes the whole process of data input, training and testing of the model. It requires the use of a function interface written in the following files. 

##### 	│  evaluate.py

​	A collection of functional interfaces written to test the accuracy of the model.

##### 	│  layer.py

​	A collection of functional interfaces written to constitute the component of the model.

##### 	│  utils.py

​	A collection of functional interfaces written for data input.

# Run This Code

First you need to execute **Dual_dangling.py** for matchable entity extraction. At this point, the program generates the extracted entity file for the corresponding datasets as 'extract_' in **Program Directory Structure**. Then, you should copy this file to the EA directory. These files will be an important part of the alignment in the second step. Then, you could run **Dual_align.py** for entity alignment and the task execution is complete. 
While for datasets with a small number of entities like **GA16K** with one-side dangling entities, you can directly execute the joint learning version of the method, as shown in **Dual_joint.py**.

# Postscript

**A.** For ablation study, you could see **layer.py** and then modify the boolean variable in **Dual_align.py** corresponding to each component needs to be melted. 

**B.** The “edge doubling” tricks could be found in **utils.py** as indicated in the program comments.

**C.** There are many parts that can be adjusted in practical applications. If the required performance is not achieved, consider the following cases:

1. Consider whether the proportion of positive samples in this scenario is too small that it is necessary to add additional weight to the loss function by improving the “beta” in **Dual_dangling.py**.
2. Consider whether it is necessary to use edge doubling trick in the dangling detection phase, because we found in the experiment that over-reliance on introducing more relational information is not always beneficial for entity classification.
3. When the accuracy of alignment is ideal, you can choose the bootstrap method for data enhancement for further better performance. We give a simpler version of bootstrap in **Dual_align.py** in the end of the code of the program.

# Acknowledgement

The datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align), [JAPE](https://github.com/nju-websoft/JAPE), and [RSNs](https://github.com/nju-websoft/RSN) and [GAKG](https://github.com/davendw49/gakg). 

1. ent_ids_1: ids for entities in source KG;
2. ent_ids_2: ids for entities in target KG;
3. ref_ent_ids: entity links encoded by ids;
4. triples_1: relation triples encoded by ids in source KG;
5. triples_2: relation triples encoded by ids in target KG;

We refer to the codes of these repos: [keras-gat](https://github.com/danielegrattarola/keras-gat), [GCN-Align](https://github.com/1049451037/GCN-Align) and [Dual-AMN](https://github.com/MaoXinn/Dual-AMN). Thanks for their great contributions!

# Environment

1. Python = 3.6
2. Keras = 2.2.5
3. Tensorflow = 1.14.0
4. Scipy
5. Numpy
6. tqdm
7. numba
