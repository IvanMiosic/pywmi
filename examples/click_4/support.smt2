(set-logic QF_UF)
(declare-fun sim_0 () Bool)
(declare-fun sim_1 () Bool)
(declare-fun sim_2 () Bool)
(declare-fun sim_3 () Bool)
(declare-fun cl_0_0 () Bool)
(declare-fun cl_0_1 () Bool)
(declare-fun cl_1_0 () Bool)
(declare-fun cl_1_1 () Bool)
(declare-fun cl_2_0 () Bool)
(declare-fun cl_2_1 () Bool)
(declare-fun cl_3_0 () Bool)
(declare-fun cl_3_1 () Bool)
(declare-fun b_0_0 () Bool)
(declare-fun b_0_1 () Bool)
(declare-fun b_1_0 () Bool)
(declare-fun b_1_1 () Bool)
(declare-fun b_2_0 () Bool)
(declare-fun b_2_1 () Bool)
(declare-fun b_3_0 () Bool)
(declare-fun b_3_1 () Bool)
(assert (let ((.def_0 (not sim_3))) (let ((.def_1 (and .def_0 b_3_1))) (let ((.def_2 (and sim_3 b_3_0))) (let ((.def_3 (or .def_2 .def_1))) (let ((.def_4 (= cl_3_1 .def_3))) (let ((.def_5 (= cl_3_0 b_3_0))) (let ((.def_6 (and .def_5 .def_4))) (let ((.def_7 (not sim_2))) (let ((.def_8 (and .def_7 b_2_1))) (let ((.def_9 (and sim_2 b_2_0))) (let ((.def_10 (or .def_9 .def_8))) (let ((.def_11 (= cl_2_1 .def_10))) (let ((.def_12 (= cl_2_0 b_2_0))) (let ((.def_13 (and .def_12 .def_11))) (let ((.def_14 (not sim_1))) (let ((.def_15 (and .def_14 b_1_1))) (let ((.def_16 (and sim_1 b_1_0))) (let ((.def_17 (or .def_16 .def_15))) (let ((.def_18 (= cl_1_1 .def_17))) (let ((.def_19 (= cl_1_0 b_1_0))) (let ((.def_20 (and .def_19 .def_18))) (let ((.def_21 (not sim_0))) (let ((.def_22 (and .def_21 b_0_1))) (let ((.def_23 (and sim_0 b_0_0))) (let ((.def_24 (or .def_23 .def_22))) (let ((.def_25 (= cl_0_1 .def_24))) (let ((.def_26 (= cl_0_0 b_0_0))) (let ((.def_27 (and .def_26 .def_25))) (let ((.def_28 (and .def_27 .def_20 .def_13 .def_6))) .def_28))))))))))))))))))))))))))))))
(check-sat)
