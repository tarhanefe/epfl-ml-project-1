import numpy as np


class Preprocess:

    def __init__(self, x_train, x_test, y_train, train_ids, test_ids):
        """
        x_train: numpy array, the training data
        x_test: numpy array, the testing data
        y_train: numpy array, the training labels
        train_ids: numpy array, the training IDs
        test_ids: numpy array, the testing IDs
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.train_ids = train_ids
        self.test_ids = test_ids

        self.set_headers()
        self.set_removal()

    def set_removal(self):
        self.removal_list = np.array(
            [
                "_STATE",
                "FMONTH",
                "IDATE",
                "IMONTH",
                "IYEAR",
                "DISPCODE",
                "SEQNO",
                "_PSU",
                "CTELENUM",
                "PVTRESD1",
                "COLGHOUS",
                "STATERES",
                "CELLFON3",
                "LADULT",
                "NUMADULT",
                "CCLGHOUS",
                "CSTATE",
                "LANDLINE",
                "NUMPHON2",
                "CPDEMO1",
                "INTERNET",
                "DIABEDU",
                "ARTHEXER",
                "ARTHWGT",
                "ARTHEDU",
                "PCPSAAD2",
                "PCPSADI1",
                "QSTVER",
                "_DUALUSE",
                "_DUALCOR",
                "ALCDAY5",
                "FRUITJU1",
                "FRUIT1",
                "FVBEANS",
                "FVGREEN",
                "FVORANG",
                "VEGETAB1",
                "IMFVPLAC",
                "HIVTSTD3",
                "WHRTST10",
                "EXRACT11",
                "EXEROFT1",
                "EXRACT21",
                "EXEROFT2",
                "STRENGTH",
                "BLDSUGAR",
                "FEETCHK2",
                "EYEEXAM",
                "CRGVREL1",
                "CRGVPRB1",
                "CRGVMST2",
                "VINOCRE2",
                "PCDMDECN",
                "RCSRLTN2",
                "CASTHDX2",
                "CASTHNO2",
                "QSTVER",
                "QSTLANG",
                "_LLCPWT",
                "_DUALCOR",
                "_DUALUSE",
                "CTELNUM1",
                "CELLFON2",
            ]
        )

    def set_headers(self):
        """
        This method sets the `column_headers` attribute to a numpy array containing
        the names of the columns for the dataset. These headers are used to label
        the data columns in the dataset for further processing and analysis.
        Attributes:
        -----------
        column_headers : numpy.ndarray
            An array of strings representing the names of the columns in the dataset.
        """
        self.column_headers = np.array(
            [
                "_STATE",
                "FMONTH",
                "IDATE",
                "IMONTH",
                "IDAY",
                "IYEAR",
                "DISPCODE",
                "SEQNO",
                "_PSU",
                "CTELENUM",
                "PVTRESD1",
                "COLGHOUS",
                "STATERES",
                "CELLFON3",
                "LADULT",
                "NUMADULT",
                "NUMMEN",
                "NUMWOMEN",
                "CTELNUM1",
                "CELLFON2",
                "CADULT",
                "PVTRESD2",
                "CCLGHOUS",
                "CSTATE",
                "LANDLINE",
                "HHADULT",
                "GENHLTH",
                "PHYSHLTH",
                "MENTHLTH",
                "POORHLTH",
                "HLTHPLN1",
                "PERSDOC2",
                "MEDCOST",
                "CHECKUP1",
                "BPHIGH4",
                "BPMEDS",
                "BLOODCHO",
                "CHOLCHK",
                "TOLDHI2",
                "CVDSTRK3",
                "ASTHMA3",
                "ASTHNOW",
                "CHCSCNCR",
                "CHCOCNCR",
                "CHCCOPD1",
                "HAVARTH3",
                "ADDEPEV2",
                "CHCKIDNY",
                "DIABETE3",
                "DIABAGE2",
                "SEX",
                "MARITAL",
                "EDUCA",
                "RENTHOM1",
                "NUMHHOL2",
                "NUMPHON2",
                "CPDEMO1",
                "VETERAN3",
                "EMPLOY1",
                "CHILDREN",
                "INCOME2",
                "INTERNET",
                "WEIGHT2",
                "HEIGHT3",
                "PREGNANT",
                "QLACTLM2",
                "USEEQUIP",
                "BLIND",
                "DECIDE",
                "DIFFWALK",
                "DIFFDRES",
                "DIFFALON",
                "SMOKE100",
                "SMOKDAY2",
                "STOPSMK2",
                "LASTSMK2",
                "USENOW3",
                "ALCDAY5",
                "AVEDRNK2",
                "DRNK3GE5",
                "MAXDRNKS",
                "FRUITJU1",
                "FRUIT1",
                "FVBEANS",
                "FVGREEN",
                "FVORANG",
                "VEGETAB1",
                "EXERANY2",
                "EXRACT11",
                "EXEROFT1",
                "EXERHMM1",
                "EXRACT21",
                "EXEROFT2",
                "EXERHMM2",
                "STRENGTH",
                "LMTJOIN3",
                "ARTHDIS2",
                "ARTHSOCL",
                "JOINPAIN",
                "SEATBELT",
                "FLUSHOT6",
                "FLSHTMY2",
                "IMFVPLAC",
                "PNEUVAC3",
                "HIVTST6",
                "HIVTSTD3",
                "WHRTST10",
                "PDIABTST",
                "PREDIAB1",
                "INSULIN",
                "BLDSUGAR",
                "FEETCHK2",
                "DOCTDIAB",
                "CHKHEMO3",
                "FEETCHK",
                "EYEEXAM",
                "DIABEYE",
                "DIABEDU",
                "CAREGIV1",
                "CRGVREL1",
                "CRGVLNG1",
                "CRGVHRS1",
                "CRGVPRB1",
                "CRGVPERS",
                "CRGVHOUS",
                "CRGVMST2",
                "CRGVEXPT",
                "VIDFCLT2",
                "VIREDIF3",
                "VIPRFVS2",
                "VINOCRE2",
                "VIEYEXM2",
                "VIINSUR2",
                "VICTRCT4",
                "VIGLUMA2",
                "VIMACDG2",
                "CIMEMLOS",
                "CDHOUSE",
                "CDASSIST",
                "CDHELP",
                "CDSOCIAL",
                "CDDISCUS",
                "WTCHSALT",
                "LONGWTCH",
                "DRADVISE",
                "ASTHMAGE",
                "ASATTACK",
                "ASERVIST",
                "ASDRVIST",
                "ASRCHKUP",
                "ASACTLIM",
                "ASYMPTOM",
                "ASNOSLEP",
                "ASTHMED3",
                "ASINHALR",
                "HAREHAB1",
                "STREHAB1",
                "CVDASPRN",
                "ASPUNSAF",
                "RLIVPAIN",
                "RDUCHART",
                "RDUCSTRK",
                "ARTTODAY",
                "ARTHWGT",
                "ARTHEXER",
                "ARTHEDU",
                "TETANUS",
                "HPVADVC2",
                "HPVADSHT",
                "SHINGLE2",
                "HADMAM",
                "HOWLONG",
                "HADPAP2",
                "LASTPAP2",
                "HPVTEST",
                "HPLSTTST",
                "HADHYST2",
                "PROFEXAM",
                "LENGEXAM",
                "BLDSTOOL",
                "LSTBLDS3",
                "HADSIGM3",
                "HADSGCO1",
                "LASTSIG3",
                "PCPSAAD2",
                "PCPSADI1",
                "PCPSARE1",
                "PSATEST1",
                "PSATIME",
                "PCPSARS1",
                "PCPSADE1",
                "PCDMDECN",
                "SCNTMNY1",
                "SCNTMEL1",
                "SCNTPAID",
                "SCNTWRK1",
                "SCNTLPAD",
                "SCNTLWK1",
                "SXORIENT",
                "TRNSGNDR",
                "RCSGENDR",
                "RCSRLTN2",
                "CASTHDX2",
                "CASTHNO2",
                "EMTSUPRT",
                "LSATISFY",
                "ADPLEASR",
                "ADDOWN",
                "ADSLEEP",
                "ADENERGY",
                "ADEAT1",
                "ADFAIL",
                "ADTHINK",
                "ADMOVE",
                "MISTMNT",
                "ADANXEV",
                "QSTVER",
                "QSTLANG",
                "MSCODE",
                "_STSTR",
                "_STRWT",
                "_RAWRAKE",
                "_WT2RAKE",
                "_CHISPNC",
                "_CRACE1",
                "_CPRACE",
                "_CLLCPWT",
                "_DUALUSE",
                "_DUALCOR",
                "_LLCPWT",
                "_RFHLTH",
                "_HCVU651",
                "_RFHYPE5",
                "_CHOLCHK",
                "_RFCHOL",
                "_LTASTH1",
                "_CASTHM1",
                "_ASTHMS1",
                "_DRDXAR1",
                "_PRACE1",
                "_MRACE1",
                "_HISPANC",
                "_RACE",
                "_RACEG21",
                "_RACEGR3",
                "_RACE_G1",
                "_AGEG5YR",
                "_AGE65YR",
                "_AGE80",
                "_AGE_G",
                "HTIN4",
                "HTM4",
                "WTKG3",
                "_BMI5",
                "_BMI5CAT",
                "_RFBMI5",
                "_CHLDCNT",
                "_EDUCAG",
                "_INCOMG",
                "_SMOKER3",
                "_RFSMOK3",
                "DRNKANY5",
                "DROCDY3_",
                "_RFBING5",
                "_DRNKWEK",
                "_RFDRHV5",
                "FTJUDA1_",
                "FRUTDA1_",
                "BEANDAY_",
                "GRENDAY_",
                "ORNGDAY_",
                "VEGEDA1_",
                "_MISFRTN",
                "_MISVEGN",
                "_FRTRESP",
                "_VEGRESP",
                "_FRUTSUM",
                "_VEGESUM",
                "_FRTLT1",
                "_VEGLT1",
                "_FRT16",
                "_VEG23",
                "_FRUITEX",
                "_VEGETEX",
                "_TOTINDA",
                "METVL11_",
                "METVL21_",
                "MAXVO2_",
                "FC60_",
                "ACTIN11_",
                "ACTIN21_",
                "PADUR1_",
                "PADUR2_",
                "PAFREQ1_",
                "PAFREQ2_",
                "_MINAC11",
                "_MINAC21",
                "STRFREQ_",
                "PAMISS1_",
                "PAMIN11_",
                "PAMIN21_",
                "PA1MIN_",
                "PAVIG11_",
                "PAVIG21_",
                "PA1VIGM_",
                "_PACAT1",
                "_PAINDX1",
                "_PA150R2",
                "_PA300R2",
                "_PA30021",
                "_PASTRNG",
                "_PAREC1",
                "_PASTAE1",
                "_LMTACT1",
                "_LMTWRK1",
                "_LMTSCL1",
                "_RFSEAT2",
                "_RFSEAT3",
                "_FLSHOT6",
                "_PNEUMO2",
                "_AIDTST3",
            ]
        )

    def process(self,PCA=False,PCA_EV=0.9):
        """
        Process both training and testing datasets by performing a series of preprocessing steps.
        This method performs the following operations:
        1. Handles continuous features by arranging them appropriately.
        2. Handles categorical features by arranging them appropriately.
        3. Changes certain values to NaN that are realized late in the process.
        4. Maps certain values that are realized late in the process.
        5. Handles one-hot features by dividing them into multiple columns.
        6. Removes columns that are deemed not useful.
        7. Concatenates additional columns to the training and testing data.
        8. Imputes missing values using the mean strategy.
        Returns:
            None
        """
        # Step 1: Handle continuous features
        self.handle_continuous()  # Adjust the necessary values with NaN for continuous features
        self.y_train = self.map_labels_to_binary(self.y_train)
        self.handle_categorical()  # Adjust the necessary values with NaN for categorical features
        self.change2na_late_realized()
        self.map_late_realized()
        self.handle_onehot()  # Convert categorical features into one-hot encoded columns
        self.remove_features()  # Remove unhelpful columns from both datasets
        self.impute_values()
        self.remove_std0()
        if PCA:
            self.PCA(var_threshold=PCA_EV)
        self.x_train = np.concatenate((self.x_train, self.train_added_columns), axis=1)
        self.x_test = np.concatenate((self.x_test, self.test_added_columns), axis=1)
        self.remove_high_nan_columns(0.5)
        self.impute_values()  # Fill NaN values with the mean of the column
        self.detect_onehot_columns()
        self.standardize_data()
        #self.remove_highly_correlated_columns()
        self.add_bias_column()
        self.get_balanced_oversample()
        print(np.isnan(self.x_train).any())



    def change2na(self, keys, target_value):
        """
        Replace specified values in both x_train and x_test with NaN.

        Parameters:
        keys (list): List of column names to be processed.
        target_value (int or str): The value to be replaced with NaN.

        Notes:
        - This function modifies x_train and x_test in place.
        - If a column name in keys is not found in self.column_headers, it prints a message indicating the column was not found.
        """
        for col_name in keys:
            if col_name in self.column_headers:
                col_index = np.where(self.column_headers == col_name)[0][0]
                self.x_train[self.x_train[:, col_index] == target_value, col_index] = (
                    np.nan
                )
                self.x_test[self.x_test[:, col_index] == target_value, col_index] = (
                    np.nan
                )
            else:
                print("Column not found:", col_name)
    def remove_std0(self):
        std = np.nanstd(self.x_train, axis=0)
        columns_to_drop = np.where(std == 0)[0]
        self.x_train = np.delete(self.x_train, list(columns_to_drop), axis=1)

    def PCA(self,var_threshold=0.95):
        print("variancethreshold = {var_threshold }")
        
        standardized_data = (self.x_train - self.x_train.mean(axis = 0)) / self.x_train.std(axis = 0)
        covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        order_of_importance = np.argsort(eigenvalues)[::-1] 

        # utilize the sort order to sort eigenvalues and eigenvectors
        sorted_eigenvalues = eigenvalues[order_of_importance]
        sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
        explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        
        sum_explained_variance = 0
        for i in range(len(explained_variance)):
            sum_explained_variance += explained_variance[i]
            if sum_explained_variance >= var_threshold:
                k=i
                break
        k = i # select the number of principal components
        self.x_train = np.matmul(standardized_data, sorted_eigenvectors[:,:k]) # transform the original data
        total_explained_variance = sum(explained_variance[:k])
        print(f"Total explained variance: {total_explained_variance}")
        print(f"Number of principal components: {k+1}")
        self.x_test = np.matmul(self.x_test, sorted_eigenvectors[:,:k]) # transform the original data
        
    def get_info(self):
        print("Old Column Headers", self.column_headers)
        print(
            "New Column Headers:",
            self.new_column_headers,
            "+",
            self.added_columns.shape[1],
        )

    def remove_features(self):
        columns_to_keep = ~np.isin(self.column_headers, self.removal_list)
        self.new_column_headers = self.column_headers[columns_to_keep]
        self.x_train = self.x_train[:, columns_to_keep]
        self.x_test = self.x_test[:, columns_to_keep]  # Also apply to x_test

    def remove_high_nan_columns(self, threshold=0.7):
        """
        Removes columns from x_train and x_test that contain more than the specified threshold of NaN values.

        Parameters:
        - threshold: float, the percentage of NaN values in a column to trigger removal (default is 0.9 for 90%).

        Notes:
        - This function removes the same columns in both x_train and x_test for consistency.
        """
        # Calculate the proportion of NaN values in each column of x_train
        nan_proportion = np.mean(np.isnan(self.x_train), axis=0)

        # Identify columns where the proportion of NaNs is greater than the threshold
        columns_to_keep = nan_proportion <= threshold

        # Retain only the columns that meet the threshold in both x_train and x_test
        self.x_train = self.x_train[:, columns_to_keep]
        self.x_test = self.x_test[:, columns_to_keep]

        # Also update column headers if applicable
        return None

    def impute_values(self):
        # Calculate column-wise mean of x_train, ignoring NaNs
        col_means = np.nanmean(self.x_train, axis=0)
        col_stds = np.nanstd(self.x_train, axis=0)
        col_stds = np.nanstd(self.x_train, axis=0)

        # Impute NaN values in x_train using random Gaussian values based on the mean and std
        # Impute NaN values in x_train using random Gaussian values based on the mean and std
        inds_train = np.where(np.isnan(self.x_train))
        self.x_train[inds_train] = np.random.normal(
            loc=np.take(col_means, inds_train[1]),
            scale=np.take(col_stds, inds_train[1]),
        )
        self.x_train[inds_train] = np.random.normal(
            loc=np.take(col_means, inds_train[1]),
            scale=np.take(col_stds, inds_train[1]),
        )

        # Impute NaN values in x_test using random Gaussian values based on the column means and stds from x_train
        # Impute NaN values in x_test using random Gaussian values based on the column means and stds from x_train
        inds_test = np.where(np.isnan(self.x_test))
        self.x_test[inds_test] = np.random.normal(
            loc=np.take(col_means, inds_test[1]), scale=np.take(col_stds, inds_test[1])
        )
        self.x_test[inds_test] = np.random.normal(
            loc=np.take(col_means, inds_test[1]), scale=np.take(col_stds, inds_test[1])
        )

        return None

    def change2na_late_realized(self):
        array1 = np.array(
            [
                28,
                29,
                49,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                78,
                79,
                80,
                87,
                91,
                92,
                93,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                103,
                104,
                107,
                108,
            ]
        )
        list1 = [
            88,
            88,
            [98, 99],
            9,
            [88, 99],
            [77, 99],
            [7, 9],
            [7777, 9999],
            [7777, 9999],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [77, 99],
            [7, 9],
            [77, 99],
            88,
            [77, 99],
            [7, 9],
            88,
            [777, 999],
            [777, 999],
            [7, 9],
            [7, 9],
            [7, 9],
            [77, 99],
            [7, 8, 9],
            [7, 9],
            [777777, 999999],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
        ]
        array2 = np.array(
            [
                109,
                112,
                113,
                114,
                116,
                117,
                118,
                120,
                121,
                123,
                124,
                126,
                127,
                128,
                129,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
            ]
        )
        list2 = [
            9,
            [77, 99],
            [77, 99],
            [77, 99],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [5, 6, 7, 8],
            [5, 6, 7],
            [7, 9],
            [7, 8, 9],
            [7, 8, 9],
            7,
            7,
            7,
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
        ]
        array3 = np.array(
            [
                140,
                141,
                142,
                143,
                144,
                145,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                155,
                156,
                157,
                158,
                159,
                160,
                161,
            ]
        )
        list3 = [
            [7, 9],
            [7, 9],
            [7, 9],
            [777, 999],
            [7, 9],
            [98, 99],
            7,
            98,
            98,
            [98, 99],
            [777, 999],
            [7, 9],
            7,
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
        ]
        array4 = np.array(
            [
                162,
                163,
                164,
                165,
                166,
                167,
                168,
                169,
                170,
                171,
                172,
                173,
                174,
                175,
                176,
                177,
                178,
                179,
                180,
                181,
                182,
                183,
                184,
                185,
                186,
                187,
                188,
                189,
                190,
                192,
                193,
                194,
                195,
                196,
                197,
                198,
                199,
                200,
                204,
                205,
                206,
                207,
                208,
                209,
                210,
                211,
                212,
                213,
                214,
                215,
                218,
                223,
                224,
                225,
            ]
        )
        list4 = [
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [77, 99],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 9],
            [7, 8, 9],
            [7, 8, 9],
            [7, 9],
            [97, 99],
            [7, 9],
            [97, 99],
            [7, 9],
            [7, 9],
            9,
            [7, 9],
            [7, 9],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [7, 9],
            [7, 9],
            5,
            9,
            [77, 99],
            [77, 99],
            99,
            [7, 9],
            [97, 99],
            [7, 9],
            [7, 9],
            9,
            [7, 9],
            [7, 9],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [77, 99],
            [7, 9],
            [77, 99],
            5,
        ]

        array5 = np.array([28, 29, 30, 31, 32, 34, 35, 33])
        list5 = [[77, 99], [77, 99], [7, 9], [7, 9], [7, 9], [7, 9], [7, 9], 9]
        array6 = [36, 37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        array7 = [
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            311,
            313,
            314,
            315,
            316,
            317,
            318,
            319,
            320,
        ]
        for i in array6:
            self.x_train[self.x_train[:, i] == 7, i] = np.nan
            self.x_test[self.x_test[:, i] == 7, i] = np.nan
            self.x_train[self.x_train[:, i] == 9, i] = np.nan
            self.x_test[self.x_test[:, i] == 9, i] = np.nan
        for i in array7:
            self.x_train[self.x_train[:, i] == 9, i] = np.nan
            self.x_test[self.x_test[:, i] == 9, i] = np.nan

        self.late_realized_func(array1, list1)
        self.late_realized_func(array2, list2)
        self.late_realized_func(array3, list3)
        self.late_realized_func(array4, list4)

    def late_realized_func(self, columns, values):
        """
        Replace specified values in the specified columns with NaN for both x_train and x_test.
        """
        # Loop through each column and replace values for both datasets
        for i in range(len(columns)):
            col_index = columns[i]
            if isinstance(values[i], list):
                # If values are a list, replace all in the list
                for numorsymbol in values[i]:
                    self.x_train[
                        self.x_train[:, col_index] == numorsymbol, col_index
                    ] = np.nan
                    self.x_test[self.x_test[:, col_index] == numorsymbol, col_index] = (
                        np.nan
                    )
            else:
                # Replace single value
                self.x_train[self.x_train[:, col_index] == values[i], col_index] = (
                    np.nan
                )
                self.x_test[self.x_test[:, col_index] == values[i], col_index] = np.nan

    def map_late_realized(self):
        mapping_list = [
            79,
            112,
            113,
            114,
            118,
            145,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            195,
            197,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            31,
        ]
        map_dict = [
            [88, 0],
            [88, 0],
            [88, 98, 0],
            [88, 0],
            [8, 1],
            [97, 10],
            [88, 0],
            [88, 0],
            [88, 0],
            [888, 0],
            [8, 0],
            [8, 0],
            [8, 0],
            [8, 0],
            [98, 0],
            [98, 0],
            [88, 0],
            [88, 0],
            [88, 0],
            [88, 0],
            [88, 0],
            [88, 0],
            [88, 0],
            [88, 0],
            [3, 0],
        ]
        for i in range(len(mapping_list)):
            column_index = mapping_list[i]
            for j in range(len(map_dict[i]) - 1):
                self.x_train[
                    self.x_train[:, column_index] == map_dict[i][j], column_index
                ] = map_dict[i][-1]
                self.x_test[
                    self.x_test[:, column_index] == map_dict[i][j], column_index
                ] = map_dict[i][-1]

    def handle_categorical(self):
        keys = np.array(
            [
                "GENHLTH",
                "_RFHLTH",
                "_HCVU651",
                "_RFHYPE5",
                "_CHOLCHK",
                "_RFCHOL",
                "_LTASTH1",
                "_CASTHM1",
                "_ASTHMS1",
                "_DRDXAR1",
                "_HISPANC",
                "_AGE65YR",
                "_RFBMI5",
                "_RFSMOK3",
                "DRNKANY5",
                "_RFBING5",
                "_RFDRHV5",
                "_FRTLT1",
                "_VEGLT1",
                "_TOTINDA",
                "_PAINDX1",
                "_PA30021",
                "_PASTRNG",
                "_PASTAE1",
                "_RFSEAT2",
                "_RFSEAT3",
            ]
        )
        self.change2na(keys, 9)
        keys = np.array(["HHADULT", "PHYSHLTH"])
        self.change2na(keys, 77)
        self.change2na(keys, 88)
        self.change2na(keys, 99)
        self.change2na(np.array(["GENHLTH"]), 7)
        self.change2na(np.array(["GENHLTH"]), 9)

    def handle_continuous(self):
        # Define the keys and their corresponding numeric placeholders
        keys_array = np.array(
            [
                "_AGEG5YR",
                "HTIN4",
                "HTM4",
                "WTKG3",
                "DROCDY3_",
                "_DRNKWEK",
                "MAXVO2_",
                "FC60_",
            ]
        )
        numeric_parts_with_dot_array = np.array(
            [14.0, 999.0, 999.0, 99999.0, 900.0, 99900.0, 99900.0, 99000.0],
            dtype=object,
        )

        # Find the indices of the columns matching keys in `self.column_headers`
        places_of_continuous = np.array(
            [
                (
                    np.where(self.column_headers == value)[0][0]
                    if value in self.column_headers
                    else -1
                )
                for value in keys_array
            ]
        )

        # Iterate over each index and replace values in both x_train and x_test
        for i in range(len(places_of_continuous)):
            col_idx = places_of_continuous[i]
            # Replace specified values with NaN in x_train
            changed_rows_train = (
                self.x_train[:, col_idx] == numeric_parts_with_dot_array[i]
            )
            self.x_train[changed_rows_train, col_idx] = np.nan

            # Replace specified values with NaN in x_test
            changed_rows_test = (
                self.x_test[:, col_idx] == numeric_parts_with_dot_array[i]
            )
            self.x_test[changed_rows_test, col_idx] = np.nan

    def handle_onehot(self):
        keys = np.array(["_PRACE1", "_MRACE1"])
        self.change2na(keys, 99)
        self.change2na(keys, 77)
        keys = np.array(["_RACE", "_RACEG21", "_RACEGR3", "_RACE_G1"])
        self.change2na(keys, 9)
        keys = np.array(
            [
                "_PRACE1",
                "_MRACE1",
                "_RACE",
                "_RACEG21",
                "_RACEGR3",
                "_RACE_G1",
                "_AGEG5YR ",
                "PCPSARS1",
                "PCPSADE1",
                "SCNTPAID",
                "SCNTLPAD",
                "SXORIENT",
                "TRNSGNDR",
                "_CRACE1",
                "_CPRACE",
                "EMPLOY1",
            ]
        )
        no_columns = np.array([8, 8, 8, 8, 8, 8, 13, 5, 4, 4, 4, 4, 4, 7, 7, 8])
        self.train_added_columns = np.zeros((self.x_train.shape[0], np.sum(no_columns)))
        self.test_added_columns = np.zeros((self.x_test.shape[0], np.sum(no_columns)))
        j = 0  # dummy1
        for i in range(len(keys)):
            # print("num of features to handle in oneshot",len(keys))

            k = j + no_columns[i]  # dummy2
            col_name = keys[i]
            self.train_added_columns[:, range(j, k)] = self.create_onehot_features(
                col_name, no_columns[i], self.x_train
            )
            self.test_added_columns[:, range(j, k)] = self.create_onehot_features(
                col_name, no_columns[i], self.x_test
            )
            j = k
        pass

    def create_onehot_features(self, column_name, num_columns, dataset):
        """
        Creates one-hot encoded features for a given column.

        Parameters:
        column_name (str): Name of the column to encode.
        num_columns (int): Number of binary columns to create.
        dataset (np.array): The dataset (x_train or x_test).

        Returns:
        np.array: One-hot encoded features or None if the column is not found.
        """
        if column_name not in self.column_headers:
            print(f"Column not found: {column_name}")
            return None

        col_index = np.where(self.column_headers == column_name)[0][0]
        encoded_features = np.zeros((dataset.shape[0], num_columns))
        column_values = dataset[:, col_index]

        for i, value in enumerate(column_values):
            if np.isnan(value):
                encoded_features[i, :] = np.nan
            else:
                index = int(value) - 1
                if 0 <= index < num_columns:
                    encoded_features[i, index] = 1
                else:
                    print(
                        f"Invalid one-hot index {index} for column {column_name}. Expected max index: {num_columns - 1}"
                    )

        return encoded_features
    def detect_onehot_columns(self):
        """
        Detects columns in x_train that only contain binary values (1s and 0s)
        and adds their indices to the list `one_hot_indices`.
        """
        self.one_hot_indices = []
        for i in range(self.x_train.shape[1]):
            # Check if the column only contains 1s, 0s, or NaNs
            column = self.x_train[:, i]
            if np.all((column == 1) | (column == 0) | np.isnan(column)):
                self.one_hot_indices.append(i)

    def map_labels_to_binary(self, y_data):
        """
        Maps the labels from -1 and 1 to 0 and 1.

        Parameters:
        y_data (numpy array): The array containing labels (-1, 1)

        Returns:
        numpy array: The modified array with labels (0, 1)
        """
        # Map -1 to 0 and 1 remains unchanged
        return np.where(y_data == -1, 0, 1)

    def standardize_data(self):
        """
        Standardizes x_train and x_test using the mean and standard deviation of x_train,
        excluding one-hot encoded columns detected by `detect_onehot_columns`.

        Returns:
        None
        """
        # Detect one-hot encoded columns before standardization
        self.detect_onehot_columns()

        # Identify indices of non-one-hot columns
        non_one_hot_indices = [
            i for i in range(self.x_train.shape[1]) if i not in self.one_hot_indices
        ]

        # Calculate mean and standard deviation of non-one-hot columns
        mean = np.mean(self.x_train[:, non_one_hot_indices], axis=0)
        std = np.std(self.x_train[:, non_one_hot_indices], axis=0)
        std[std == 0] = 1  # Prevent division by zero

        # Standardize only non-one-hot columns
        self.x_train[:, non_one_hot_indices] = (
            self.x_train[:, non_one_hot_indices] - mean
        ) / std
        self.x_test[:, non_one_hot_indices] = (
            self.x_test[:, non_one_hot_indices] - mean
        ) / std

    def remove_highly_correlated_columns(self, threshold=0.9):
        """
        Removes columns with high correlation from both train and test sets.

        Args:
        threshold (float): The correlation threshold to identify highly correlated columns.

        Modifies:
        self.X_train: Updates the training set with removed columns.
        self.X_test: Updates the testing set with removed columns.
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(self.x_train, rowvar=False)
        corr_matrix = np.abs(corr_matrix)  # Use absolute values for correlation

        # Identify columns to remove based on the upper triangle
        upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
        correlated_pairs = [
            (i, j)
            for i, j in zip(*upper_triangle_indices)
            if corr_matrix[i, j] > threshold
        ]

        # Track columns to drop
        columns_to_drop = set()
        for i, j in correlated_pairs:
            # Add the higher-index column to drop list to avoid reshaping issues
            columns_to_drop.add(j)

        # Remove columns from train and test sets
        self.x_train = np.delete(self.x_train, list(columns_to_drop), axis=1)
        self.x_test = np.delete(self.x_test, list(columns_to_drop), axis=1)

    def shuffle_data(self):
        """
        Shuffles the rows of X_train and y_train consistently.

        Args:
        X_train (np.ndarray): Feature matrix.
        y_train (np.ndarray): Label vector.

        Returns:
        tuple: Shuffled X_train and y_train.
        """
        assert (
            self.x_train.shape[0] == self.y_train.shape[0]
        ), "X_train and y_train must have the same number of rows."

        # Generate a random permutation of row indices
        permutation = np.random.permutation(self.x_train.shape[0])

        # Apply permutation to both X_train and y_train
        X_train_shuffled = self.x_train[permutation]
        y_train_shuffled = self.y_train[permutation]

        return X_train_shuffled, y_train_shuffled

    def add_bias_column(self):
        # Add a column of ones to the left of x_train and x_test
        self.x_train = np.hstack((np.ones((self.x_train.shape[0], 1)), self.x_train))
        self.x_test = np.hstack((np.ones((self.x_test.shape[0], 1)), self.x_test))

    import numpy as np

    def get_balanced_oversample(self):
        """
        Returns a balanced dataset by oversampling the minority class to match the majority class count.

        Args:
            X (numpy array): Feature matrix.
            y (numpy array): Label vector (binary classification).

        Returns:
            tuple: Oversampled X and y with balanced classes.
        """
        # Identify indices for each class
        class_0_indices = np.where(self.y_train == 0)[0]
        class_1_indices = np.where(self.y_train == 1)[0]
        
        # Determine the size of the majority class
        max_class_count = max(len(class_0_indices), len(class_1_indices))
        
        # Oversample the minority class
        if len(class_0_indices) < max_class_count:
            sampled_class_0_indices = np.random.choice(class_0_indices, max_class_count, replace=True)
            balanced_indices = np.concatenate([sampled_class_0_indices, class_1_indices])
        else:
            sampled_class_1_indices = np.random.choice(class_1_indices, max_class_count, replace=True)
            balanced_indices = np.concatenate([class_0_indices, sampled_class_1_indices])
        
        np.random.shuffle(balanced_indices)  # Shuffle for random distribution
        
        # Update the training data with the balanced, oversampled dataset
        self.x_train = self.x_train[balanced_indices]
        self.y_train = self.y_train[balanced_indices]
        
        # Optionally, return the balanced dataset
