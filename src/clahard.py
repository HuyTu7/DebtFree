from __future__ import print_function, division

import matplotlib.pyplot as plt
import time
import re
import pandas as pd
import pdb
import copy
from clami import *

class CLAHard(object):
    def __init__(self,data,target,cla=False,thres=90,CLAfilter=False,CLAlabel=True,hybrid=True,toggle=False,early_stop=False):
        self.uncertain_thres = 10
        self.target = target
        self.data = copy.deepcopy(data)
        self.rest = copy.deepcopy(self.data)
        self.easy = Easy_CLAMI(copy.deepcopy(data),self.target,thres)
        self.easy.preprocess(cla=cla)
        self.hybrid = hybrid
        self.CLAfilter = CLAfilter
        self.CLAlabel = CLAlabel
        self.toggle = toggle
        self.early_stop = early_stop
        print("*" * 20)
        print("Hybrid : ", self.hybrid, " | CLA Filtering : ", self.CLAfilter,
              " | CLA Labelling : ", self.CLAlabel,
              " | Toggling : ", self.toggle, " | Early Stopping :", self.early_stop)
        print("*" * 20)
        self.stats_satd = {"actual_td": [], "estimated_td": []}

    def find_patterns(self):
        self.easy.find_patterns()

    def test_patterns_clami(self,output = False, include=False):
        self.easy.test_patterns(output=output)

        if not include:
            train_data, test_data = CLAMI_data(self.data, self.target, None, 50, label="Label")
            self.include = np.array([])
            self.rest[self.target] = test_data
            self.rest[self.target].rename(columns = {"Label":"label"}, inplace=True)
            self.rest[self.target]['label'] = np.array(["yes" if x == 1 else "no"
                                                        for x in self.rest[self.target]['label']])
            rest = {self.target: self.rest[self.target]}
            rest["train"] = train_data.rename(columns = {"Label":"label"})
            rest["train"]['label'] = np.array(["yes" if x == 1 else "no"
                                                for x in rest["train"]['label']])
            self.easy.left_train = np.array(range(rest["train"].shape[0]))
            self.rest = rest
        else:
            full_data = range(self.data[self.target].shape[0])
            self.include = np.setdiff1d(full_data, self.easy.left_test)
        print("EASY: %s" % self.easy.stats_test)
        return self.easy.stats_test

    def test_patterns(self,output = False, include=False):
        if self.CLAfilter:
            self.easy.test_patterns(output=output)
        if not include:
            self.include = np.array([])
            self.rest[self.target] = self.data[self.target].iloc[self.easy.left_test]
            if self.CLAlabel:
                start = 0
                for project in self.data:
                    if project != self.target:
                        end = len(self.data[project]["label"]) + start
                        self.rest[project]['label'] = self.easy.x_label[start:end]
                        start = end
        else:
            full_data = range(self.data[self.target].shape[0])
            self.include = np.setdiff1d(full_data, self.easy.left_test)
        print("EASY: %s" % self.easy.stats_test)
        return self.easy.stats_test


    def easy_code(self):
        content = []
        for target in self.data:
            content += [str(c).lower() for c in self.data[target]["Abstract"]]
        csr_mat = self.easy.tfer.transform(content)

        indices = {}
        for pattern in self.easy.patterns:
            try:
                id = self.easy.voc.tolist().index(pattern)
            except:
                print(pattern, " is not the pattern in the file")
            indices[pattern] = [i for i in range(csr_mat.shape[0]) if csr_mat[i,id] > 0]
        easy_code = ["no"]*len(content)

        for pattern in self.easy.patterns:
            for i in indices[pattern]:
                easy_code[i]="yes"
        start = 0
        for project in self.data:
            end = len(self.data[project]["label"]) + start
            self.data[project]["easy_code"] = easy_code[start:end]
            self.rest[project]=self.data[project][self.data[project]["easy_code"]=="no"]
            start=end

    def output_target(self, path):
        target_easy = self.data[self.target][self.data[self.target]["easy_code"] == "yes"]
        target_rest = self.rest[self.target]
        target_easy.to_csv(path + self.target + "_easy.csv", line_terminator="\r\n", index=False)
        target_rest.to_csv(path + self.target + "_rest.csv", line_terminator="\r\n", index=False)

    def output_conflicts(self,output="../new_data/conflicts/"):
        for project in self.data:
            x = self.data[project]
            conflicts = x[x["easy_code"]=="yes"][x["label"]=="no"]
            conflicts.to_csv(output+project+".csv", line_terminator="\r\n", index=False)

    def apply_hard(self, model = "RF", est = False):
        self.hard = Hard(model=model, est=est)
        self.hard.create(self.rest, self.target)

    def query_hard(self, tmp = "../httpd/http_query.csv", output = "../httpd/httpd_rest.csv", batch_size = 10):
        try:
            coded = pd.read_csv(tmp)
            for i in range(len(coded)):
                self.hard.body["code"].loc[:self.hard.newpart-1][self.hard.body["ID"][:self.hard.newpart] == coded["ID"][i]] = coded["code"][i]
                self.hard.body["time"].loc[:self.hard.newpart-1][self.hard.body["ID"][:self.hard.newpart] == coded["ID"][i]] = time.time()
        except:
            pass
        pos, neg, total = self.hard.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, self.hard.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))
        if pos + neg >= total:
            return
        a, b, c, d = self.hard.train()
        if pos < self.uncertain_thres:
            sample = a[:batch_size]
        else:
            sample = c[:batch_size]
        self.hard.body.loc[sample].to_csv(tmp, line_terminator="\r\n", index=False)
        df_yes = self.hard.body.loc[:self.hard.newpart-1].loc[self.hard.body["code"][:self.hard.newpart]=="yes"]
        df_no = self.hard.body.loc[:self.hard.newpart-1].loc[self.hard.body["code"][:self.hard.newpart]=="no"]
        df_ignore = self.hard.body.loc[:self.hard.newpart-1].loc[self.hard.body["code"][:self.hard.newpart]=="undetermined"]
        pd.concat([df_yes,df_no,df_ignore], ignore_index=True).to_csv(output, line_terminator="\r\n", index=False)


    def ML_hard(self, model = "RF", est = False, T_rec = 0.9):
        start_early = True if self.toggle else False
        self.hard = Hard(model=model, est=est, inc=self.include, start_early=start_early)
        self.hard.create(self.rest, self.target, important=self.easy.left_train)
        step = 100 if not self.toggle else 10
        lives = 5
        prev_pos, prev_neg, prev_total = 0, 0, 0
        one_time_toggle = False if self.toggle else True
        hybrid_est = .1
        hybrid_flag = False
        # pdb.set_trace()
        while True:
            pos, neg, total = self.hard.get_numbers()
            if pos + neg >= total:
                break
            a, b, c, d = self.hard.train()
            print("Current Pos=%s Neg=%s Total=%s Estimated=%s" % (pos, neg, total, self.hard.est_num))
            self.stats_satd["actual_td"].append(pos)
            self.stats_satd["estimated_td"].append(self.hard.est_num)
            ratio = float(pos) / (pos + neg) if (pos + neg) > 0 else 0
            if self.early_stop and float(pos - prev_pos) / step <= 0.1 \
                    and self.hard.est_num > 0 and pos >= self.hard.est_num * 0.5:
                lives -= 1
                if lives == 0:
                    print("*!*" * 10)
                    print("Early Stopping")
                    print("*!*" * 10)
                    break
            else:
                lives = 5

            if self.hybrid and pos >= self.hard.est_num * hybrid_est and pos > 0 and neg > 0 and self.hard.est_num > 0 and one_time_toggle:
                step = 10
                self.hard.body = self.hard.body[:self.hard.newpart]
                print("*!*" * 10)
                print("Drop Training Data")
                print("*!*" * 10)

            if self.toggle and pos > 20 and pos < 100 and ratio <= 0.1 and not one_time_toggle:
                self.hard.extra_body["code"][:self.hard.newpart] = self.hard.body["code"]
                self.hard.body = self.hard.extra_body.copy()
                print("*!*" * 10)
                print("Unsupervised Labelling the Training Data")
                print("*!*" * 10)
                self.hard.toggle = True
                one_time_toggle = True

            if self.hard.est_num>0 and pos >= self.hard.est_num*T_rec and pos > total * 0.02:
                if self.hybrid and T_rec < 1:
                    break
                elif not self.hybrid:
                    break

            if pos < self.uncertain_thres:
                self.hard.code_batch(a[:step])
            else:
                if step >= 100:
                    step = int(float(step)/2)
                self.hard.code_batch(c[:step])
            prev_pos = pos
        return self.hard

    def eval(self):
        stat = Counter(self.data[self.target]['label'])
        t = stat["yes"]
        n = stat["no"]
        order = np.argsort(self.hard.body["time"][:self.hard.newpart])
        tp = self.easy.stats_test['tp']
        fp = self.easy.stats_test['p'] - tp
        tn = n - fp
        fn = t - tp

        # for stopping at target recall
        if self.include.size > 0:
            hard_tp = self.hard.record['pos'][-1] - self.include.shape[0]
            hard_p = self.hard.record['x'][-1] - self.include.shape[0]
        else:
            hard_tp = self.hard.record['pos'][-1]
            hard_p = self.hard.record['x'][-1]
        all_tp = hard_tp+tp
        all_fp = fp+hard_p-hard_tp
        prec = all_tp / float(all_fp+all_tp)
        rec = all_tp / float(t)
        f1 = 2*prec*rec/(prec+rec)
        fall_out = float(all_fp) / (all_fp + tn)
        g1 = (2 * rec * (1 - fall_out)) / (rec + 1 - fall_out)
        last_cost=hard_p/(t+n)
        ######################

        cost = 0
        costs = [cost]
        tps = [tp]
        fps = [fp]
        tns = [tn]
        fns = [fn]

        for label in self.hard.body["label"][order]:
            cost+=1.0
            costs.append(cost)
            if label=="yes":
                tp+=1.0
                fn-=1.0
            else:
                fp+=1.0
                tn-=1.0
            fps.append(fp)
            tps.append(tp)
            tns.append(tn)
            fns.append(fn)

        costs = np.array(costs)
        tps = np.array(tps)
        fps = np.array(fps)
        tns = np.array(tns)
        fns = np.array(fns)

        tpr = tps / (tps+fns)
        fpr = fps / (fps+tns)
        costr = costs / (t+n)

        auc = self.AUC(tpr,fpr)
        apfd = self.AUC(tpr,costr)

        return {"AUC":auc, "APFD":apfd, "TPR":tpr, "CostR":costr, "FPR":fpr, "Precision": prec, "Recall": rec, "F1": f1, "G1": g1, "Cost": last_cost}

    def AUC(self,ys,xs):
        assert len(ys)==len(xs), "Size must match."
        if type(xs)!=type([]):
            xs=list(xs)
        if type(ys)!=type([]):
            ys=list(ys)
        x_last = 0
        if xs[-1]<1.0:
            xs.append(1.0)
            ys.append(ys[-1])
        auc = 0.0
        for i,x in enumerate(xs):
            y = ys[i]
            auc += y*(x-x_last)
            x_last = x
        return auc



class Hard(object):
    def __init__(self,model="RF", est=False, inc=np.array([]), start_early=False):
        self.step = 10
        self.enable_est = est
        self.include = inc
        self.model_name = ""
        self.extra_body = None
        self.toggle = False
        self.start_early = start_early
        if model=="RF":
            self.model = RandomForestClassifier(class_weight="balanced_subsample")
        elif model=="NB":
            self.model = MultinomialNB()
        elif model == "LR":
            self.model = LogisticRegression(class_weight="balanced")
        elif model == "DT":
            self.model = DecisionTreeClassifier(class_weight="balanced",max_depth=8)
        elif model == "SVM":
            self.model = SGDClassifier(class_weight="balanced")
        elif model == "CNN":
            self.model_name = "CNN"

    def create(self, data, target, important=np.array([])):
        # pdb.set_trace()
        self.record = {"x":[], "pos":[], 'est':[]}
        self.body = {}
        self.est = []
        self.est_num = 0
        self.target = target
        if self.model_name == "CNN":
            tm = Treatment(data, target)
            tm.get_data()
            self.body = pd.DataFrame(0, index=np.arange(len(tm.x_label)), columns=["Abstract","label","code","time"])
            tmp = pd.DataFrame(0, index=np.arange(len(tm.y_label)), columns=["Abstract","label","code","time"])
            self.body["Abstract"], self.body["label"] = load_data_jb(tm.x_content, tm.x_label)
            tmp["Abstract"], tmp["label"] = load_data_jb(tm.y_content, tm.y_label)
            tmp['code'] = ["undetermined"] * len(tmp)
            self.body['code'] = self.body['label']
            self.body = pd.concat([tmp, self.body], ignore_index=True)
            self.newpart = len(tm.y_label)
        else:
            self.loadfile(data[target])
            self.create_old(data, important=important)
            # self.csr_mat = self.body.values[:, :-3]
            self.preprocess()
        return self

    def loadfile(self, data):
        self.body = data
        self.body['code'] = ["undetermined"]*len(self.body)
        self.body['time'] = [0.0]*len(self.body)
        if self.start_early:
            while True:
                self.body = self.body.sample(frac=1).reset_index(drop=True)
                label_sample = self.body['label'][:100]
                y = label_sample[label_sample == "yes"].shape[0]
                if y != 100 and y != 0:
                    self.body['code'][:100] = label_sample
                    print("Have both categories in the data's labels!")
                    break
        self.newpart = len(self.body)
        return

    ### Use previous knowledge, labeled only
    def create_old(self,data,important=np.array([])):
        # pdb.set_trace()
        bodies = []
        for key in data:
            if key == self.target:
                continue
            body = data[key]
            label = body["label"]
            body['code'] = pd.Series(label)
            bodies.append(body)
        if important.size == 0:
            bodies = [self.body.copy()] + bodies
            self.extra_body = pd.concat(bodies, ignore_index=True)
        else:
            if self.include.size == 0:
                body = pd.concat(bodies, ignore_index=True)
                body = body.iloc[important]
                self.extra_body = pd.concat([self.body, body], ignore_index=True)
            else:
                self.body["code"].iloc[self.include] = "yes"
                bodies = [self.body.copy()] + bodies
                self.extra_body = pd.concat(bodies, ignore_index=True)
        if not self.start_early:
            self.body = self.extra_body.copy()
        print("Train Data Starting at", self.body.shape, self.extra_body.shape)

    def get_numbers(self):
        total = len(self.body["code"][:self.newpart])
        pos = Counter(self.body["code"][:self.newpart])["yes"]
        neg = Counter(self.body["code"][:self.newpart])["no"]

        try:
            tmp = self.record['x'][-1]
        except:
            tmp = -1

        if int(pos+neg) > tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
            self.record['est'].append(int(self.est_num))

        self.pool = np.where(np.array(self.body['code'][:self.newpart]) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code'][:self.newpart]))) - set(self.pool))
        return pos, neg, total


    def preprocess(self):
        sample = np.unique(np.concatenate([np.where(np.array(self.body['code']) != "undetermined")[0],
                                           np.where(np.array(self.extra_body['code']) != "undetermined")[0]]))
        # if not self.start_early:
        #     self.extra_body["Abstract"] = self.extra_body["Abstract"].astype(str)
        #     content0, content = self.extra_body["Abstract"][self.newpart:], self.extra_body["Abstract"]
        # else:
        #     self.body["Abstract"] = self.body["Abstract"].astype(str)
        #     content0, content = self.body["Abstract"][:100], self.body["Abstract"]
        self.extra_body["Abstract"] = self.extra_body["Abstract"].astype(str)
        content0, content = self.extra_body["Abstract"][sample], self.extra_body["Abstract"]
        tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                               sublinear_tf=False, decode_error="ignore")
        tfer.fit(content0)
        self.csr_mat = tfer.transform(content)
        # self.voc = np.array(list(tfer.vocabulary_.keys()))[np.argsort(list(tfer.vocabulary_.values()))]

        return

    ## Train model ##
    def train(self):
        # pdb.set_trace()
        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        # pdb.set_trace()
        # labels = np.array(["yes" if x == 1 else "no" for x in self.body["code"][sample]])
        if self.toggle:
            self.extra_body["Abstract"] = self.extra_body["Abstract"].astype(str)
            content0, content = self.extra_body["Abstract"][sample], self.extra_body["Abstract"]
            tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                                   sublinear_tf=False, decode_error="ignore")
            tfer.fit(content0)
            self.csr_mat = tfer.transform(content)
            self.toggle = False
        self.model.fit(self.csr_mat[sample], self.body["code"][sample])
        # new_sample = sample[np.argsort(self.body['time'][sample])[::-1][:self.step]]
        # self.model.partial_fit(self.csr_mat[new_sample], self.body["code"][new_sample])

        if self.enable_est:
            self.est_num, self.est = self.estimate_curve()

        uncertain_id, uncertain_prob = self.uncertain()
        certain_id, certain_prob = self.certain()

        return uncertain_id, uncertain_prob, certain_id, certain_prob

        ## Get uncertain ##

    def train_CNN(self):
        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        self.decisions, self.probs = train_model(self.body["Abstract"][sample], self.body["code"][sample], self.target,
                                                allow_save_model=True, d_lossweight=0.0, model_path="best_hard")

        uncertain_id, uncertain_prob = self.uncertain_cnn()
        certain_id, certain_prob = self.certain_cnn()
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    def uncertain_cnn(self):
        pos_at = list(self.decisions).index(1.0)
        prob = self.probs[self.pool][:, pos_at]
        order = np.argsort(np.abs(prob - 0.5))
        return np.array(self.pool)[order], np.array(prob)[order]


    def certain_cnn(self):
        pos_at = list(self.decisions).index(1.0)
        prob = self.probs[self.pool][:, pos_at]
        order = np.argsort(prob)[::-1]
        return np.array(self.pool)[order], np.array(prob)[order]

    def uncertain(self):
        pos_at = list(self.model.classes_).index("yes")
        if type(self.model).__name__ == "SGDClassifier":
            prob = self.model.decision_function(self.csr_mat[self.pool])
            order = np.argsort(np.abs(prob))
        else:
            prob = self.model.predict_proba(self.csr_mat[self.pool])[:, pos_at]
            order = np.argsort(np.abs(prob-0.5))
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get certain ##
    def certain(self):
        pos_at = list(self.model.classes_).index("yes")
        if type(self.model).__name__ == "SGDClassifier":
            prob = self.model.decision_function(self.csr_mat[self.pool])
            order = np.argsort(prob)
            if pos_at>0:
                order = order[::-1]
        else:
            prob = self.model.predict_proba(self.csr_mat[self.pool])[:, pos_at]
            order = np.argsort(prob)[::-1]
        return np.array(self.pool)[order],np.array(self.pool)[order]


    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)


    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: self.body[key][i] for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result


    ## Code candidate studies ##
    def code(self,id,label):
        self.body["code"][id] = label
        self.body["time"][id] = time.time()


    def code_batch(self,ids):
        # hpdb.set_trace()
        now = time.time()
        times = [now+id/10000000.0 for id in range(len(ids))]
        labels = self.body["label"][ids]
        self.body["code"][ids] = labels
        self.body["time"][ids] = times

    def estimate_curve(self):

        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= 1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count -= 1
                    can = []
            return sample

        ###############################################
        clf = LogisticRegression(penalty='l2', fit_intercept=True, class_weight="balanced")
        sample = np.where(np.array(self.body['code']) != "undetermined")[0]
        # labels = np.array(["yes" if x == 1 else "no" for x in self.body["code"][sample]])
        clf.fit(self.csr_mat[sample], self.body["code"][sample])

        prob = clf.decision_function(self.csr_mat[:self.newpart])
        prob2 = np.array([[p] for p in prob])

        y = np.array([1 if x == 'yes' else 0 for x in self.body['code'][:self.newpart]])
        y0 = np.copy(y)

        all = range(len(y))

        pos_num_last = Counter(y0)[1]
        if pos_num_last<10:
            return 0, []
        pos_origin = pos_num_last
        old_pos = pos_num_last - Counter(self.body["code"][:self.newpart])["yes"]

        lifes = 1
        life = lifes

        while (True):
            C = pos_num_last / pos_origin
            es = LogisticRegression(penalty='l2', fit_intercept=True, C=1)
            es.fit(prob2[all], y[all])
            pos_at = list(es.classes_).index(1)
            pre = es.predict_proba(prob2[self.pool])[:, pos_at]

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            pos_num = Counter(y)[1]

            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num

        esty = pos_num - old_pos
        pre = es.predict_proba(prob2)[:, pos_at]

        return esty, pre

    def get_allpos(self):
        return Counter(self.body["label"][:self.newpart])["yes"]

    def plot(self, T_rec = 0.9):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        t = float(self.get_allpos())
        n = float(len(self.body["label"][:self.newpart]))

        fig = plt.figure()

        costr = np.array(self.record["x"])/n
        tpr = np.array(self.record["pos"])/t
        estr = np.array(self.record["est"])/t

        plt.plot(costr, tpr,label='Recall')
        plt.plot(costr, estr,'--',label='Estimation')
        plt.plot(costr, [1.0]*len(tpr),'-.',label='100% Recall')
        plt.plot(costr, [T_rec]*len(tpr),':',label=str(int(T_rec*100))+'% Recall')
        plt.grid()

        plt.ylim(0,1.5)
        plt.legend()
        plt.xlabel("Cost")
        plt.savefig("../figures_est/" + self.target + ".png")
        plt.close(fig)
        print(self.target)
        print((tpr[-1],costr[-1]))


class Easy(object):
    def __init__(self,data,target,thres=0.9):
        self.data=data
        self.target=target
        self.thres = thres
        self.stats_test = {'tp':0,'p':0}
        self.x_content = []
        self.x_label = []

        for project in data:
            if project==target:
                continue
            self.x_content += [str(c).lower() for c in data[project]["Abstract"]]
            self.x_label += [c for c in data[project]["label"]]

        self.x_label = np.array(self.x_label)
        self.y_content = [str(c).lower() for c in data[target]["Abstract"]]
        self.y_label = [c for c in data[target]["label"]]

    def preprocess(self):

        self.tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                               sublinear_tf=False, decode_error="ignore")
        self.train_data = self.tfer.fit_transform(self.x_content)
        self.test_data = self.tfer.transform(self.y_content)
        self.voc = np.array(list(self.tfer.vocabulary_.keys()))[np.argsort(list(self.tfer.vocabulary_.values()))]

    def find_patterns(self):
        left_train = range(self.train_data.shape[0])
        self.pattern_ids = []
        self.precs = []
        self.train_data[self.train_data.nonzero()]=1
        while True:
            id, fitness = self.get_top_fitness(self.train_data[left_train],self.x_label[left_train])
            left_train,stats_train = self.remove(self.train_data,self.x_label,left_train, id)
            prec = float(stats_train["tp"])/stats_train["p"]
            if prec < self.thres - 0.1:
                break
            self.pattern_ids.append(id)
            self.precs.append(prec)
        self.patterns = self.voc[self.pattern_ids]


    def get_top_fitness(self,matrix,label):
        poses = np.where(label=="yes")[0]
        count_tp = np.array(np.sum(matrix[poses],axis=0))[0]
        count_p = np.array(np.sum(matrix,axis=0))[0]

        fitness = np.nan_to_num(count_tp*(count_tp/count_p)**3)
        order = np.argsort(fitness)[::-1]
        top_fitness = count_tp[order[0]]/count_p[order[0]]

        print({'tp':count_tp[order[0]], 'fp':count_p[order[0]]-count_tp[order[0]], 'fitness':top_fitness})
        return order[0], top_fitness

    def remove(self, data, label, left, id):
        to_remove = set()
        p = 0
        tp = 0
        for row in left:
            if data[row,id]>0:
                to_remove.add(row)
                p+=1
                if label[row]=="yes":
                    tp+=1

        left = list(set(left)-to_remove)
        return left, {"p":p, "tp":tp}

    def test_patterns(self,output=False):
        left_test = range(self.test_data.shape[0])
        self.stats_test={"tp":0,"p":0}
        for id in self.pattern_ids:
            left_test,stats_test = self.remove(self.test_data,self.y_label,left_test, id)
            self.stats_test["tp"]+=stats_test["tp"]
            self.stats_test["p"]+=stats_test["p"]
        # save the "hard to find" data
        if output:
            self.rest = self.data[self.target].loc[left_test]
            self.rest.to_csv("../new_data/rest/csc/"+self.target+".csv", line_terminator="\r\n", index=False)
        return self.stats_test


class MAT(Easy):
    def find_patterns(self):
        self.patterns = ["todo","fixme","hack","xxx"]
        for x in self.patterns:
            try:
                self.pattern_ids = [self.voc.tolist().index(x)]
            except:
                print(x, " is not the pattern in the file")

class MAT_Two_Step(CLAHard):
    def __init__(self,data,target):
        self.uncertain_thres = 0
        self.target = target
        self.data = data
        self.rest = self.data.copy()
        self.easy = MAT(self.data,self.target)
        self.easy.preprocess()

    def ML_hard(self, model = "RF"):
        treatments = {"RF":RF,"SVM":SVM,"LR":LR,"NB":NB,"DT":DT,"TM":TM}
        self.hard = treatments[model](self.rest,self.target)
        self.hard.preprocess()
        self.hard.train()
        return self.hard

    def eval(self):
        stat = Counter(self.data[self.target]['label'])
        t = stat["yes"]
        n = stat["no"]
        order = np.argsort(self.hard.probs)[::-1]
        tp = self.easy.stats_test['tp']
        fp = self.easy.stats_test['p'] - tp
        tn = n - fp
        fn = t - tp
        cost = 0
        costs = [cost]
        tps = [tp]
        fps = [fp]
        tns = [tn]
        fns = [fn]

        for label in np.array(self.hard.y_label)[order]:
            cost+=1.0
            costs.append(cost)
            if label=="yes":
                tp+=1.0
                fn-=1.0
            else:
                fp+=1.0
                tn-=1.0
            fps.append(fp)
            tps.append(tp)
            tns.append(tn)
            fns.append(fn)

        costs = np.array(costs)
        tps = np.array(tps)
        fps = np.array(fps)
        tns = np.array(tns)
        fns = np.array(fns)

        tpr = tps / (tps+fns)
        fpr = fps / (fps+tns)
        costr = costs / (t+n)

        auc = self.AUC(tpr,fpr)
        apfd = self.AUC(tpr,costr)

        return {"AUC":auc, "APFD":apfd, "TPR":tpr, "CostR":costr, "FPR":fpr}


class Easy_Two_Step(MAT_Two_Step):
    def __init__(self,data,target):
        self.uncertain_thres = 0
        self.target = target
        self.data = data
        self.rest = self.data.copy()
        self.easy = Easy(self.data,self.target)
        self.easy.preprocess()

class Easy_CLAMI(object):
    def __init__(self, data, target, thres=0.05):
        self.data = data
        self.target = target
        self.thres = thres
        self.num_data = {}
        self.x_label = []
        self.stats_test = {'tp': 0, 'p': 0}

    def preprocess(self, cla=False):
        treatment = Treatment(self.data, self.target)
        treatment.preprocess()
        testdata = treatment.full_test
        traindata = treatment.full_train
        # pdb.set_trace()
        start = 0
        for project in self.data:
            if project != self.target:
                end = len(self.data[project]["label"]) + start
                tmp = getInstancesByCLA(traindata[start:end], self.thres, None)
                self.num_data[project] = tmp
                self.x_label.append(["yes" if y == 1 else "no" for y in self.num_data[project]["CLA"]])
                start = end
        # self.train_data = getInstancesByCLA(traindata, self.thres, None)
        self.cla = cla
        if not cla:
            self.x_label = np.array(treatment.x_label)
        else:
            print("do unsupervised learning labels")
            self.x_label = np.concatenate(self.x_label)

        self.test_data = getInstancesByCLA(testdata, 50, None)
        self.y_label = np.array(["yes" if y == 1 else "no" for y in self.test_data["Label"]])
        self.left_train = np.array(range(self.x_label.shape[0]))
        self.left_test = np.array(range(self.test_data.shape[0]))

    def find_patterns(self):
        # self.thresholds = [np.percentile(self.train_data['K'], 50 + x*5) for x in range(1, 9, 1)]
        # self.thresholds = [50 + (x * 5) for x in range(1, 9, 1)]
        self.thresholds = [55]
        self.best_thres = -1
        self.left_train = []
        worst_thres = -1
        best_prec = -1
        worst_prec = 1

        for t in self.thresholds:
            start = 0
            for project in self.data:
                if project != self.target:
                    end = len(self.data[project]["label"]) + start
                    left_train = range(end - start)
                    thres = np.percentile(self.num_data[project]['K'], t)
                    left_train, stats_train = self.remove_data(self.num_data[project], self.x_label[start:end],
                                                               left_train, thres, start_index=start)
                    print(t, stats_train, left_train.shape)
                    self.left_train.append(left_train)
                    start = end
        self.best_thres = 55
        self.left_train = np.concatenate(self.left_train)
        # self.left_train, _ = self.remove_train(self.train_data, self.x_label,
        #                                        range(self.train_data.shape[0]), worst_thres)
        # pdb.set_trace()
        # self.train_label = self.train_label.iloc[self.left_train]
        all_p = np.count_nonzero(np.array(self.x_label) == "yes")
        # self.left_train = np.array(range(self.train_data.shape[0]))
        print("Removing train instances: %s" % (stats_train["tp"] / all_p))
        # print("Removing instances: %s" % (self.left_train.shape[0] / self.train_data.shape[0]))
        print("Best Threshold : %s,  Best Prec : %s" % (self.best_thres, best_prec))
        print("Worst Threshold : %s,  Worst Prec : %s" % (worst_thres, worst_prec))

    def find_patterns_tmp(self):
        # self.thresholds = [np.percentile(self.train_data['K'], 50 + x*5) for x in range(1, 9, 1)]
        # self.thresholds = [50 + (x * 5) for x in range(1, 9, 1)]
        self.thresholds = [55]
        self.best_thres = -1
        worst_thres = -1
        best_prec = -1
        worst_prec = 1

        for t in self.thresholds:
            left_train = range(self.train_data.shape[0])
            thres = np.percentile(self.train_data['K'], t)
            left_train, stats_train = self.remove_data(self.train_data, self.x_label, left_train, thres)
            print(t, stats_train)
            prec = float(stats_train["tp"]) / (stats_train["p"] + 0.00001)
            if prec > best_prec:
                stats_train = stats_train
                self.left_train = left_train
                best_prec = prec
                self.best_thres = t
            if prec < worst_prec:
                worst_prec = prec
                worst_thres = thres
        # self.left_train, _ = self.remove_train(self.train_data, self.x_label,
        #                                        range(self.train_data.shape[0]), worst_thres)
        # self.train_label = self.train_label.iloc[self.left_train]
        all_p = np.count_nonzero(np.array(self.x_label) == "yes")
        self.left_train = np.array(range(self.train_data.shape[0]))
        print("Removing train instances: %s" % (stats_train["tp"] / all_p))
        # print("Removing instances: %s" % (self.left_train.shape[0] / self.train_data.shape[0]))
        print("Best Threshold : %s,  Best Prec : %s" % (self.best_thres, best_prec))
        print("Worst Threshold : %s,  Worst Prec : %s" % (worst_thres, worst_prec))

    def remove_train(self, data, label, left, thres):
        K = data['K'].values
        p_arr = np.where(K > thres)[0]
        n_arr = np.where(K < thres)[0]
        tn_arr = np.where(label == "no")[0]
        tp_arr = np.where(label == "yes")[0]
        tp_arr = np.intersect1d(p_arr, tp_arr)
        tn_arr = np.intersect1d(n_arr, tn_arr)
        p = p_arr.shape[0]
        tp = tp_arr.shape[0]
        # for row in left:
        #     if data.iloc[row]['K'] > thres:
        #         to_remove.add(row)
        #         p += 1
        #         if label[row] == "yes":
        #             tp += 1

        #left = list(set(left) - to_remove)
        left = np.setdiff1d(left, tp_arr)
        left = np.setdiff1d(left, tn_arr)

        return left, {"p": p, "fp": tp, "fn": fn}


    def remove_data(self, data, label, left, thres, start_index=0):
        K = data['K'].values
        p_arr = np.where(K > thres)[0]
        tp_arr = np.where(label == "yes")[0]
        tp_arr = np.intersect1d(p_arr, tp_arr)
        tp_arr = [x[0] for x in sorted(zip(tp_arr, K[tp_arr]), key=lambda x: x[1])]
        tp_arr = np.array(tp_arr)[:int(len(tp_arr) * 0.9)]

        n_arr = np.where(K <= thres)[0]
        tn_arr = np.where(label == "no")[0]
        tn_arr = np.intersect1d(n_arr, tn_arr)
        tn_arr = [x[0] for x in sorted(zip(tn_arr, K[tn_arr]), key=lambda x: x[1])]
        tn_arr = np.array(tn_arr)[-int(len(tn_arr) * 0.9):]

        left = np.setdiff1d(left, tp_arr)
        left = np.setdiff1d(left, tn_arr) + start_index

        return left, {"p": p_arr.shape[0], "tp": tp_arr.shape[0]}

    def remove(self, data, label, left, thres, start_index=0):
        K = data['K'].values
        np.argsort(K)
        p_arr = np.where(K > thres)[0]
        tp_arr = np.where(label == "yes")[0]
        tp_arr = np.intersect1d(p_arr, tp_arr)
        p = p_arr.shape[0]
        tp = tp_arr.shape[0]
        # for row in left:
        #     if data.iloc[row]['K'] > thres:
        #         to_remove.add(row)
        #         p += 1
        #         if label[row] == "yes":
        #             tp += 1

        # left = list(set(left) - to_remove)
        # p_arr = np.random.choice(p_arr, int(p_arr.shape[0] * .9), replace=False)
        left = np.setdiff1d(left, p_arr) + start_index

        return left, {"p": p, "tp": tp}

    def test_patterns(self,output=False):
        self.left_test = range(self.test_data.shape[0])
        self.stats_test = {"tp": 0, "p": 0}
        #
        self.best_thres = 95
        thres = np.percentile(self.test_data['K'], self.best_thres)
        # thres = np.percentile(self.train_data['K'], self.best_thres)
        self.left_test, stats_test = self.remove(self.test_data, self.y_label, self.left_test, thres)
        self.stats_test["tp"] += stats_test["tp"]
        self.stats_test["p"] += stats_test["p"]
        # save the "hard to find" data

        all_p = np.count_nonzero(np.array(self.y_label) == "yes")
        # print("Removing instances: %s" % (self.stats_test["tp"] / self.stats_test["p"]))
        print("Removing test instances: %s" % (self.stats_test["tp"] / all_p))
        if output:
            self.rest = self.data[self.target].loc[left_test]
            self.rest.to_csv("../new_data/rest/csc/"+self.target+".csv", line_terminator="\r\n", index=False)
        return self.stats_test