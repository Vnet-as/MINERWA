FROM grafana/grafana:10.2.0

COPY ./provisioning/plugins/clickhouse.yaml /etc/grafana/provisioning/plugins/clickhouse.yaml
COPY ./provisioning/datasources/clickhouse.yaml /etc/grafana/provisioning/datasources/clickhouse.yaml
COPY ./provisioning/dashboards/minerwa.yaml /etc/grafana/provisioning/dashboards/minerwa.yaml
COPY ./provisioning/dashboards/dashboards /var/lib/grafana/dashboards

RUN grafana-cli plugins install grafana-clickhouse-datasource
