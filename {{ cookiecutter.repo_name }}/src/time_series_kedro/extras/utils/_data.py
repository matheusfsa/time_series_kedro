class TSDataset:
    def __init__(self, df, serie_id, serie_target, serie_date, serie_group, serie_exogs):
        
        self.df = df
        self.serie_id = serie_id
        self.serie_target = serie_target
        self.serie_date = serie_date 
        self.serie_group = serie_group 
        self.serie_exogs = serie_exogs if isinstance(serie_exogs, list) or serie_exogs is None else [serie_exogs,]
        
        self.ids = iter(self.df[serie_id].unique())
        self.idx = 0
        
    def __iter__(self):
        for serie_id in self.ids:
            data = self.df[self.df[self.serie_id] == serie_id].set_index(self.serie_date)
            group = data[self.serie_group].iloc[0]
            if self.serie_exogs is not None:
                X = data[self.serie_exogs]
            else:
                X = None
            y = data[self.serie_target]
            yield serie_id, group, X, y
        