from Confluence_page_classification.util import Util

class MlProcessing:
    def __init__(self):
        self.prev_data = None
        self.step = 0
        self.data = None
        pass

    def set_step(self, step):
        self.step = step

    def export_step(self, step=None):
        if step:
            self.step = step

        if self.prev_data is not None:
            Util.write_delta_dataframe_to_excel(prev_df=self.prev_data,
                                                current_df=self.data,
                                                step_number=self.step)
        self.prev_data = self.data.copy()
        self.step += 1
