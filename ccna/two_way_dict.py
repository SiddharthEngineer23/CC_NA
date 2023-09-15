import pandas as pd

class TwoWayDict(dict):
    
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2
    
    def from_dict(self, user_dict):
        for key, value in user_dict.items():
            dict.__setitem__(self, key, value)
            dict.__setitem__(self, value, key)

def _load_country_name_conversion(base_dir):
    country_name_conversion = pd.read_csv(f"{base_dir}input/wikipedia-iso-country-codes.csv")
    country_name_conversion.rename(columns={"English short name lower case": "Country Name", "Alpha-2 code": "ISO2", "Alpha-3 code": "ISO3"}, inplace=True)
    return country_name_conversion

def create_country_name_lookup(base_dir):
    country_name_conversion = _load_country_name_conversion(base_dir)
    two_way_country_lookup = TwoWayDict()
    oneway_lookup = country_name_conversion.set_index("Country Name")["ISO2"].to_dict()
    two_way_country_lookup.from_dict(oneway_lookup)
    return two_way_country_lookup