"""Run report: facts and warnings collected by each stage, written as report.md and report.json."""
import json


class Report:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.stages = {}    # stage name -> dict of facts
        self.warnings = []  # (stage, message)

    def stage(self, name, **facts):
        self.stages.setdefault(name, {}).update(facts)

    def warn(self, stage, message):
        self.warnings.append((stage, message))
        print(f'  warning [{stage}]: {message}')

    def write(self):
        data = {'stages': self.stages,
                'warnings': [{'stage': s, 'message': m} for s, m in self.warnings]}
        (self.run_dir / 'report.json').write_text(json.dumps(data, indent=2) + '\n')
        lines = [f'# tabvid report: {self.run_dir.name}', '']
        for name, facts in self.stages.items():
            lines.append(f'## {name}')
            lines += [f'- {key}: {value}' for key, value in facts.items()]
            lines.append('')
        lines.append(f'## Warnings ({len(self.warnings)})')
        lines += [f'- [{s}] {m}' for s, m in self.warnings] or ['- none']
        (self.run_dir / 'report.md').write_text('\n'.join(lines) + '\n')
