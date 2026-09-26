# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Generate a portable README for a local experiment directory."""

import os


class ReadmeGenerator:
    def __init__(self, base_url, project_folder, short_name):
        # base_url is retained for backwards API compatibility. Experiment
        # READMEs now use relative links so they remain valid off GitHub too.
        self.base_url = base_url
        self.project_folder = project_folder
        self.short_name = short_name.strip().replace(".", "")

    def write(self):
        images = [
            self.short_name + "_price.png",
            self.short_name + "_hist.png",
            self.short_name + "_prediction.png",
            "MSE.png",
            "loss.png",
        ]
        with open(os.path.join(self.project_folder, "README.md"), "w", encoding="utf-8") as handle:
            for image in images:
                handle.write(f"![]({image.replace(' ', '%20')})\n")
