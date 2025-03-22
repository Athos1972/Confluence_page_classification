import toml
import sys
import getopt


class Singleton(type):
    """
    Singleton-Klasse.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """
    Globale Konfiguration aus toml-File lesen und zur Verfügung stellen. Singleton-Klasse - wird nur 1x instanziert
    """
    def __init__(self, filename=None):
        """

        :param filename: Dateiname (inkl. Pfad - wenn notwendig) zur Konfigurationsdatei
        """
        if not filename:
            filename = self.__get_cli_args("c")
        try:
            self.config = toml.load(filename)
        except FileNotFoundError:
            sys.exit(f"Config-File {filename} not found. Aborting")

        # Die sonstigen CLI-Parameter auslesen - sie überschreiben potentielle config-Parameter
        if self.__get_cli_args("l"):
            self.config["LIMIT"] = self.__get_cli_args("l")
        if self.__get_cli_args("t"):
            self.config["TAGS"] = self.__get_cli_args("t")

    @staticmethod
    def __get_cli_args(argument_to_search_for: str = "c"):
        """
        Holt die CLI-Parameter ab.
        c = Config-File
        h = Hilfe
        l = Limit / Anzahl Seiten
        t = Tags / Confluence-Tags
        s = Spaces / Confluence-Spaces
        :return:
        """
        correct_call = "Aufruf mittels: <file>.py -c <config_file.toml> -l <limit> -t <tags> -s <spaces>"

        if argument_to_search_for not in ["c", "l", "t"]:
            print(correct_call)
            sys.exit()

        argv = sys.argv[1:]
        try:
            opts, args = getopt.getopt(argv, "c:", ["config="])
        except getopt.GetoptError:
            print(correct_call)
            sys.exit(2)

        if not opts:
            print(correct_call)
            sys.exit()

        for opt, arg in opts:
            if opt == '-h':
                print(correct_call)
                sys.exit()
            elif opt in ("-c", "--config") and argument_to_search_for == "c":
                return arg
            elif opt in ("-l", "--limit") and argument_to_search_for == "l":
                return arg
            elif opt in ("-t", "--tags") and argument_to_search_for == "t":
                return arg
            elif opt in ("-s", "--spaces") and argument_to_search_for == "s":
                return arg
            else:
                return None

    def get_config(self, config_key: str, optional=True, default_value=None):
        """
        Gibt den/die Werte aus dem Config-File (TOML) zurück

        :param config_key: Schlüssel, der aus dem Config-File rauskomen soll
        :param optional: Nicht dumpen, wenn der Parameter nicht gefunden wurde
        :param default_value: Wert, der zurückgegeben werden soll wenn nichts gefunden wurde
        :return:
        """
        if optional:
            return self.config.get(config_key, default_value)

        if config_key not in self.config.keys():
            raise ValueError(f"Key {config_key} not found in config-file and not optional. Aborting.")

        return self.config[config_key]
